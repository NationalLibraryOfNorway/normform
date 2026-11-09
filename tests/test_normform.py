"""Unit tests for the standalone NormForm package."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from normform import (
    ChoiceCache,
    DocumentHarness,
    StyleProfile,
    canonical_choice_key,
    clean_direct,
    export_sealed_cache,
    extract_changed_alternatives,
    find_inconsistencies,
    forced_choice,
    greedy_set_cover,
    load_invalid_pairs,
    load_seed_cache,
    load_style_profile,
    load_yaml,
    parse,
    render_cached_template,
    resolve,
    resolve_candidate_choice,
    split_sentences,
)
from normform.cli import main as cli_main


class FakeAdapter:
    def alternatives(
        self, text: str, *, temperature: float = 0.0, strict: bool = False
    ) -> dict[str, object]:
        mapping = {
            "Jenta hoppa høyt.": "[Jenta|Jenten] [hoppa|hoppet] høyt.",
            "Boka lå på bordet.": "[Boka|Boken] lå på bordet.",
            "Boken var ny.": "[Boka|Boken] var ny.",
            "Han så et band.": "Han så et [band|bånd].",
        }
        out = mapping.get(text, text)
        return {
            "task": "create_morphological_alternatives",
            "text": out,
            "raw_proposal": out,
            "source_copy_fallback": False,
        }

    def style(
        self,
        words: Sequence[str],
        input_text: str,
        *,
        temperature: float = 0.0,
        strict: bool = False,
    ) -> dict[str, object]:
        tmpl = parse(input_text)
        chosen = []
        for grp in tmpl.groups:
            if "jenta" in [w.lower() for w in words] and "Jenta" in grp:
                chosen.append("Jenta")
            elif "hoppa" in [w.lower() for w in words] and "hoppa" in grp:
                chosen.append("hoppa")
            else:
                chosen.append(grp[-1])
        rendered = tmpl.render(tuple(chosen))
        return {
            "task": "skriv_i_samme_morfologiske_stil",
            "text": rendered,
            "raw_proposal": rendered,
            "choices": tuple(chosen),
        }


def test_contract_parse_align_and_clean_direct() -> None:
    tmpl = parse("[Katta|Katten] ligger på [matta|matten] [1].")
    assert tmpl.groups == (("Katta", "Katten"), ("matta", "matten"))
    assert tmpl.align("Katta ligger på matten [1].") == ("Katta", "matten")

    # Prohibited group [et|ett] is automatically collapsed to source token
    cleaned = clean_direct("Jeg har et hus.", "Jeg har [et|ett] [hus|huset].")
    assert cleaned == "Jeg har et [hus|huset]."


def test_constrained_beam_resolve() -> None:
    sentence = "[Katta|Katten] ligger på [matta|matten]."
    res = resolve(
        sentence,
        proposal="Invalid literal modification",
        score=lambda cands: [1.0 if "Katten" in c and "matten" in c else 0.0 for c in cands],
        beam_size=2,
    )
    assert res.used_constrained_search is True
    assert res.text == "Katten ligger på matten."
    assert res.choices == ("Katten", "matten")


def test_yaml_loader_preserves_norwegian_no(tmp_path: Path) -> None:
    yaml_file = tmp_path / "style.yaml"
    yaml_file.write_text(
        "_metadata:\n  language: nno\n  preserve_unreviewed: true\n"
        "choices:\n  no | nå: no\n  boka | boken: boka\n"
        "invalid_pairs:\n  band | bånd: cross-role homograph\n",
        encoding="utf-8",
    )
    parsed = load_yaml(yaml_file.read_text(encoding="utf-8"))
    assert parsed["_metadata"]["preserve_unreviewed"] is True
    assert parsed["choices"]["no | nå"] == "no"

    seed = load_seed_cache(yaml_file)
    assert seed[canonical_choice_key(("no", "nå"))] == "no"
    invalid = load_invalid_pairs(yaml_file)
    assert ("band", "bånd") in invalid

    profile = load_style_profile(yaml_file, lang="nno")
    assert profile is not None
    assert profile.choices[canonical_choice_key(("boka", "boken"))] == "boka"
    assert ("band", "bånd") in profile.invalid_pairs


def test_resolve_candidate_choice_casing_and_invalid_pairs() -> None:
    invalid = {("band", "bånd"): "polysemous"}
    choice, hit = resolve_candidate_choice(("band", "bånd"), "band", "bånd", invalid)
    assert (choice, hit) == ("band", False)

    choice2, hit2 = resolve_candidate_choice(("Boka", "Boken"), "Boken", "boka", {})
    assert (choice2, hit2) == ("Boka", True)

    choice3, hit3 = resolve_candidate_choice(("vebb-TV", "web-TV"), "web-TV", "vebb-tv", {})
    assert (choice3, hit3) == ("vebb-TV", True)


def test_document_harness_with_style_profile_and_invalid_pairs(tmp_path: Path) -> None:
    yaml_file = tmp_path / "style.yaml"
    yaml_file.write_text(
        "_metadata:\n  language: nob\n  prompt_words: [jenta, hoppa]\n"
        "choices:\n  boka | boken: boka\n"
        "invalid_pairs:\n  band | bånd: homograph\n",
        encoding="utf-8",
    )
    profile = load_style_profile(yaml_file, lang="nob")
    cache = ChoiceCache(tmp_path / "cache.json", model_identifier="test-model")
    harness = DocumentHarness(
        FakeAdapter(),
        cache,
        style_profile=profile,
        preserve_unreviewed=True,
    )
    out = harness.process_record(
        {"text": "Boken var ny. Han så et band."},
        record_index=0,
        default_prompt_words=[],
    )
    # 'Boken' -> 'Boka' from style_profile; 'band' preserved due to invalid_pairs
    assert out["standardised_text"] == "Boka var ny. Han så et band."


def test_greedy_set_cover_selects_minimal_sentences() -> None:
    cand_to_sents = {
        "g1": [0, 1],
        "g2": [1, 2],
        "g3": [1],
    }
    sent_to_cands = {
        0: ["g1"],
        1: ["g1", "g2", "g3"],
        2: ["g2"],
    }
    selected = greedy_set_cover(cand_to_sents, sent_to_cands)
    assert selected == [1]


def test_audit_inconsistencies_and_stats(tmp_path: Path) -> None:
    cache = ChoiceCache(tmp_path / "cache.json", model_identifier="test-model")
    harness = DocumentHarness(FakeAdapter(), cache)
    records = [
        {"text": "Boka lå på bordet."},
        {"text": "Boken var ny."},
    ]
    issues = find_inconsistencies(records, harness, [], raw_mode=True)
    assert len(issues) == 1
    assert issues[0]["first_occurrence"]["choice"] == "Boka"
    assert issues[0]["inconsistent_occurrence"]["choice"] == "Boken"

    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text(
        '{"text": "Boken var ny.", "normalized": "Boka var ny."}\n',
        encoding="utf-8",
    )
    ranked, scanned, changed = extract_changed_alternatives(corpus_file)
    assert scanned == 1
    assert changed == 1
    assert ranked[0][1] == "boka"


def test_export_sealed_cache_and_cli_styles_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed = tmp_path / "sealed.yaml"
    decision_stats: dict[str, dict[str, object]] = {}
    tmpl = parse("[Boka|Boken] var ny.")
    rendered = render_cached_template(
        tmpl,
        "Boken var ny.",
        {canonical_choice_key(("Boka", "Boken")): "boka"},
        {},
        decision_stats,
    )
    assert rendered == "Boka var ny."
    export_sealed_cache(
        choices={canonical_choice_key(("Boka", "Boken")): "boka"},
        output_path=sealed,
        metadata={"language": "nob", "sealed_at": "2026-11-09T00:00:00Z"},
        decision_stats=decision_stats,
        invalid_pairs={("band", "bånd"): "homograph"},
    )
    assert sealed.is_file()
    loaded = load_seed_cache(sealed)
    assert loaded[canonical_choice_key(("boka", "boken"))] == "boka"

    rc = cli_main(["styles", "list"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "NbAiLab/normform-bokmaal-redaksjonsnorm" in captured
    assert "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm" in captured


def test_split_sentences_and_forced_choice() -> None:
    spans = split_sentences("Hei! Hvordan går det? Bra.")
    assert [s.text.strip() for s in spans] == ["Hei!", "Hvordan går det?", "Bra."]
    assert forced_choice(("boka", "boken"), ["Boka"]) == "boka"

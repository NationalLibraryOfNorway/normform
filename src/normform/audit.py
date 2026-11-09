"""Morphological inconsistency auditing and corpus change-impact reporting."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .contract import parse
from .harness import DocumentHarness, choice_key, split_sentences
from .styles import load_yaml


def find_inconsistencies(
    records: list[dict[str, Any]],
    harness: DocumentHarness,
    prompt_words: Sequence[str],
    *,
    use_cache: bool = True,
    raw_mode: bool = False,
) -> list[dict[str, Any]]:
    """Scan records and identify inconsistent choices for identical candidate groups."""
    history: dict[str, dict[str, Any]] = {}
    inconsistencies: list[dict[str, Any]] = []

    for rec_idx, record in enumerate(records):
        source_text = record.get("text", "")
        if not isinstance(source_text, str) or not source_text.strip():
            continue

        spans = split_sentences(source_text)

        if not use_cache:
            harness.cache.clear()

        for sent_idx, span in enumerate(spans):
            if not span.text.strip():
                continue

            if raw_mode:
                res = harness.adapter.alternatives(span.text)
                alt_text = res.get("text", span.text)
                if "[" not in alt_text or "]" not in alt_text:
                    continue
                try:
                    template = parse(alt_text)
                    if not template.groups:
                        continue
                    choices = template.align(span.text)
                    final_text = span.text
                except Exception:
                    continue
            else:
                processed = harness.process_record(
                    {"text": span.text},
                    record_index=rec_idx,
                    default_prompt_words=prompt_words,
                )
                final_text = processed.get("standardised_text", span.text)
                try:
                    template = parse(span.text)
                    if not template.groups:
                        continue
                    choices = template.align(final_text)
                except Exception:
                    continue

            for grp_idx, candidates in enumerate(template.groups):
                key = choice_key(candidates)
                choice = choices[grp_idx]

                if key not in history:
                    history[key] = {
                        "choice": choice,
                        "candidates": list(candidates),
                        "record_index": rec_idx,
                        "sentence_index": sent_idx,
                        "original_text": span.text,
                        "standardised_text": final_text,
                    }
                else:
                    prev = history[key]
                    if prev["choice"] != choice:
                        inconsistencies.append(
                            {
                                "candidate_group": list(candidates),
                                "group_key": key,
                                "first_occurrence": {
                                    "choice": prev["choice"],
                                    "record_index": prev["record_index"],
                                    "sentence_index": prev["sentence_index"],
                                    "original_text": prev["original_text"],
                                    "standardised_text": prev["standardised_text"],
                                },
                                "inconsistent_occurrence": {
                                    "choice": choice,
                                    "record_index": rec_idx,
                                    "sentence_index": sent_idx,
                                    "original_text": span.text,
                                    "standardised_text": final_text,
                                },
                            }
                        )

    return inconsistencies


def load_candidate_groups(
    cache_path: Path | None = None,
) -> tuple[
    dict[tuple[str, ...], str | None],
    dict[str, list[tuple[tuple[str, ...], str | None]]],
]:
    """Load known morphological candidate groups from a YAML/JSON cache."""
    all_groups: dict[tuple[str, ...], str | None] = {}
    word_to_groups: dict[str, list[tuple[tuple[str, ...], str | None]]] = defaultdict(
        list
    )

    if cache_path and cache_path.exists():
        content = cache_path.read_text(encoding="utf-8")
        data = (
            load_yaml(content)
            if cache_path.suffix in (".yaml", ".yml")
            else json.loads(content)
        )
        raw_choices = data.get("choices", data) if isinstance(data, dict) else {}
        for k, chosen_val in raw_choices.items():
            if str(k).startswith("_") or str(k) == "invalid_pairs":
                continue
            chosen = (
                str(
                    chosen_val.get("choice", "")
                    if isinstance(chosen_val, dict)
                    else chosen_val
                )
                .strip()
                .lower()
            )
            cands = tuple(sorted({c.strip().lower() for c in str(k).split("|")}))
            all_groups[cands] = chosen
            for c in cands:
                word_to_groups[c].append((cands, chosen))

    return all_groups, word_to_groups


def extract_changed_alternatives(
    corpus_path: Path,
    text_field: str = "text",
    normalized_field: str = "normalized",
    cache_path: Path | None = None,
    lang: str = "nob",
    limit: int | None = None,
) -> tuple[list[tuple[tuple[str, ...], str, int, list[tuple[str, int]]]], int, int]:
    """Scan corpus and aggregate change statistics per alternative group."""
    _, word_to_groups = load_candidate_groups(cache_path)
    token_re = re.compile(r"[\w]+", re.UNICODE)

    diff_counts: Counter[tuple[str, str]] = Counter()
    total_records = 0
    total_changed_records = 0

    if corpus_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(corpus_path, columns=[text_field, normalized_field])
        orig_col = table[text_field].to_pylist()
        norm_col = table[normalized_field].to_pylist()
        for orig, std in zip(orig_col, norm_col):
            total_records += 1
            if orig != std:
                total_changed_records += 1
                ot = token_re.findall(str(orig).lower())
                st = token_re.findall(str(std).lower())
                if len(ot) == len(st):
                    for o, s in zip(ot, st):
                        if o != s:
                            diff_counts[(o, s)] += 1
            if limit and total_records >= limit:
                break
    else:
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_records += 1
                obj = json.loads(line)
                orig = str(obj.get(text_field, ""))
                std = str(obj.get(normalized_field, ""))
                if orig != std:
                    total_changed_records += 1
                    ot = token_re.findall(orig.lower())
                    st = token_re.findall(std.lower())
                    if len(ot) == len(st):
                        for o, s in zip(ot, st):
                            if o != s:
                                diff_counts[(o, s)] += 1
                if limit and total_records >= limit:
                    break

    group_input_changes: dict[tuple[str, ...], Counter[tuple[str, str]]] = defaultdict(
        Counter
    )
    group_targets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for (o, s), cnt in diff_counts.items():
        matched_grp: tuple[str, ...] | None = None
        if o in word_to_groups:
            for grp, _ in word_to_groups[o]:
                if s in grp:
                    matched_grp = grp
                    break
        if not matched_grp:
            matched_grp = tuple(sorted([o, s]))

        group_input_changes[matched_grp][(o, s)] += cnt
        group_targets[matched_grp][s] += cnt

    ranked_groups = []
    for grp, pair_dict in group_input_changes.items():
        chosen = group_targets[grp].most_common(1)[0][0]
        clean_inputs: Counter[str] = Counter()
        for (o, s), cnt in pair_dict.items():
            if s.lower() == chosen.lower() and o.lower() != chosen.lower():
                clean_inputs[o] += cnt

        total_chg = sum(clean_inputs.values())
        if total_chg > 0:
            ranked_groups.append((grp, chosen, total_chg, clean_inputs.most_common()))

    ranked_groups.sort(key=lambda item: -item[2])
    return ranked_groups, total_records, total_changed_records


def extract_approximate_cache_usage(
    corpus_path: Path,
    cache_path: Path,
    text_field: str = "text",
    normalized_field: str = "normalized",
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Conservatively infer cache usage from aligned source/output word tokens."""
    all_groups, word_to_groups = load_candidate_groups(cache_path)
    token_re = re.compile(r"[\w]+", re.UNICODE)
    supported_groups = {
        group: chosen
        for group, chosen in all_groups.items()
        if chosen and all(token_re.fullmatch(candidate) for candidate in group)
    }
    usage: dict[tuple[str, ...], dict[str, Any]] = {
        group: {
            "group": group,
            "chosen": chosen,
            "seen": 0,
            "changed": 0,
            "already_correct": 0,
            "changed_inputs": Counter(),
        }
        for group, chosen in supported_groups.items()
    }
    metadata = {
        "records_scanned": 0,
        "unaligned_records_skipped": 0,
        "ambiguous_occurrences_skipped": 0,
        "non_single_token_groups_not_counted": len(all_groups) - len(supported_groups),
    }

    def process_record(orig: Any, std: Any) -> None:
        metadata["records_scanned"] += 1
        original_tokens = token_re.findall(str(orig).lower())
        normalized_tokens = token_re.findall(str(std).lower())
        if len(original_tokens) != len(normalized_tokens):
            metadata["unaligned_records_skipped"] += 1
            return
        for original, normalized in zip(original_tokens, normalized_tokens):
            matches = []
            for group, chosen in word_to_groups.get(original, []):
                if chosen and group in usage and normalized == chosen:
                    matches.append(group)
            matches = list(dict.fromkeys(matches))
            if len(matches) > 1:
                metadata["ambiguous_occurrences_skipped"] += 1
                continue
            if not matches:
                continue
            stats = usage[matches[0]]
            stats["seen"] += 1
            if original == normalized:
                stats["already_correct"] += 1
            else:
                stats["changed"] += 1
                stats["changed_inputs"][original] += 1

    if corpus_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(corpus_path, columns=[text_field, normalized_field])
        for orig, std in zip(
            table[text_field].to_pylist(), table[normalized_field].to_pylist()
        ):
            process_record(orig, std)
            if limit and metadata["records_scanned"] >= limit:
                break
    else:
        with corpus_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                process_record(
                    record.get(text_field, ""),
                    record.get(normalized_field, ""),
                )
                if limit and metadata["records_scanned"] >= limit:
                    break

    rows = list(usage.values())
    rows.sort(key=lambda item: (-int(item["seen"]), item["group"]))
    metadata["decisions_seen"] = sum(int(item["seen"]) for item in rows)
    metadata["decisions_changed"] = sum(int(item["changed"]) for item in rows)
    metadata["decisions_already_correct"] = sum(
        int(item["already_correct"]) for item in rows
    )
    return rows, metadata

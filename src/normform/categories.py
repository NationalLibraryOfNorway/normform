"""Grammatical word-class and root stem-family classification for Norwegian morphological alternatives."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CategoryClassification:
    """Grammatical or stem-family classification for a candidate group."""

    category_id: str
    category_title: str
    category_order: int
    tag: str
    stem_family: tuple[str, ...] | None = None
    is_simplex_root: bool = False


CATEGORY_META: dict[str, tuple[int, str]] = {
    "VERB_INF": (1, "VERB — Infinitive (-a / -e & -j- Alternation)"),
    "VERB_PAST_PART": (2, "VERB — Weak Preterite & Past Participle (-a / -et / -d / -de / -te)"),
    "VERB_PRES_PASS": (3, "VERB — Present, Passive & Present Participle (-er / -jer, -ast / -jast)"),
    "NOUN_FEM_DEF_SG": (4, "NOUN — Feminine / Common Definite Singular (-a / -en)"),
    "NOUN_DEF_SG": (5, "NOUN — Definite Singular (-a / -en / -et)"),
    "NOUN_NEUT_IND_PL": (6, "NOUN — Neuter Indefinite Plural (zero / -er / metathesis)"),
    "NOUN_NEUT_DEF_PL": (7, "NOUN — Neuter Definite Plural (-a / -ene)"),
    "NOUN_PL_IND": (8, "NOUN — Masculine / Feminine Indefinite Plural (-ar / -er / -r)"),
    "NOUN_PL_DEF": (9, "NOUN — Masculine / Feminine Definite Plural (-ane / -ene / -one)"),
    "ADJ_PART_PL": (10, "ADJ / PART — Definite & Plural Participles (-a / -ede / -ete)"),
    "ADJ_DET_AGR": (11, "ADJ / DET / NOUN — Gender, Indefinite & Suffix Agreement (-n / -t / -e)"),
    "LEXICAL_STEM": (12, "LEXICAL & STEM VARIANTS (Grouped by Root Family, then Frequency)"),
}


NOB_INFLECTION_MAP: dict[tuple[str, ...], tuple[str, str]] = {
    # Feminine / Common Definite Singular (-a / -en)
    ("a", "en"): ("NOUN_FEM_DEF_SG", "NOUN.Fem.Sing.Def"),
    ("a", "uen"): ("NOUN_FEM_DEF_SG", "NOUN.Fem.Sing.Def"),
    # Neuter Definite Plural (-a / -ene)
    ("a", "ene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("a", "aene", "ene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("era", "erene", "rene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("era", "rene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("la", "lene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("ela", "lene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("ma", "mene", "mma", "mmene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("ma", "mmene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    ("a", "ma", "mene"): ("NOUN_NEUT_DEF_PL", "NOUN.Neuter.Plur.Def"),
    # Neuter Indefinite Plural (zero / -er / metathesis / gemination)
    ("", "er"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("", "mer"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("", "ter"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("", "ner"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("", "rer"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("er", "re"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("er", "ere", "re"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("ere", "re", "rer"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("el", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("del", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("kel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("tel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("gel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("bel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("pel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    ("sel", "ler"): ("NOUN_NEUT_IND_PL", "NOUN.Neuter.Plur.Ind"),
    # Masculine / Feminine Plurals
    ("", "r"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("ar", "er"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("ane", "ene"): ("NOUN_PL_DEF", "NOUN.Plur.Def"),
    # Weak Preterite & Past Participle
    ("a", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "et", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "et", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "d", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "dd", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "de", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "dde", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("na", "net", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("na", "net", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("la", "let", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("la", "let", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("ra", "ret", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("ra", "ret", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("de", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("d", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("dd", "tt"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("et", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("et", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("dde", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("de", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    # Definite & Plural Participles (-a / -ede / -ete)
    ("a", "ede", "ete"): ("ADJ_PART_PL", "ADJ.PerfPart.Def/Plur"),
    ("ede", "ete"): ("ADJ_PART_PL", "ADJ.PerfPart.Def/Plur"),
    ("a", "ede"): ("ADJ_PART_PL", "ADJ.PerfPart.Def/Plur"),
    ("a", "ete"): ("ADJ_PART_PL", "ADJ.PerfPart.Def/Plur"),
    # Gender & Indefinite Agreement
    ("n", "t"): ("ADJ_DET_AGR", "ADJ/DET.Gender"),
    ("in", "n"): ("ADJ_DET_AGR", "ADJ/DET.Gender"),
    ("", "t"): ("ADJ_DET_AGR", "ADJ/DET.Neuter"),
    ("", "e"): ("ADJ_DET_AGR", "NOUN/ADJ.Suffix.-e"),
}


NNO_INFLECTION_MAP: dict[tuple[str, ...], tuple[str, str]] = {
    # Infinitive (-a / -e & -j- infinitives)
    ("a", "e"): ("VERB_INF", "VERB.Inf"),
    ("a", "e", "ja", "je"): ("VERB_INF", "VERB.Inf"),
    ("ja", "je"): ("VERB_INF", "VERB.Inf"),
    ("", "a", "e"): ("VERB_INF", "VERB.Inf"),
    ("a", "ja"): ("VERB_INF", "VERB.Inf.-j-"),
    ("e", "je"): ("VERB_INF", "VERB.Inf.-j-"),
    ("ja", "je", "ta", "te"): ("VERB_INF", "VERB.Inf"),
    ("ga", "ge", "gja", "gje"): ("VERB_INF", "VERB.Inf"),
    ("ka", "ke", "kja", "kje"): ("VERB_INF", "VERB.Inf"),
    # Present, Passive & Participle
    ("er", "jer"): ("VERB_PRES_PASS", "VERB.Pres"),
    ("ast", "jast"): ("VERB_PRES_PASS", "VERB.Pass"),
    ("ande", "jande"): ("VERB_PRES_PASS", "VERB.PresPart"),
    # Weak Preterite & Past Participle
    ("a", "et"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("de", "te"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("d", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("dd", "tt"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("a", "t"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    ("de", "ete"): ("VERB_PAST_PART", "VERB.Past/PerfPart"),
    # Indefinite & Definite Plurals
    ("ar", "er"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("ar", "mar"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("arar", "jarar"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("ar", "jar"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("", "ar"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("", "er"): ("NOUN_PL_IND", "NOUN.Plur.Ind"),
    ("ane", "ene"): ("NOUN_PL_DEF", "NOUN.Plur.Def"),
    ("ene", "one"): ("NOUN_PL_DEF", "NOUN.Fem.Plur.Def"),
    ("ane", "mane"): ("NOUN_PL_DEF", "NOUN.Plur.Def"),
    # Definite Singular
    ("a", "en"): ("NOUN_DEF_SG", "NOUN.Def.Sing"),
    ("en", "men"): ("NOUN_DEF_SG", "NOUN.Masc.Sing.Def"),
    ("aren", "jaren"): ("NOUN_DEF_SG", "NOUN.Masc.Sing.Def"),
    ("et", "tet"): ("NOUN_DEF_SG", "NOUN.Neuter.Sing.Def"),
    ("a", "na"): ("NOUN_DEF_SG", "NOUN.Fem.Sing.Def"),
    ("", "ne"): ("NOUN_DEF_SG", "NOUN.Def"),
    # ADJ / DET / PRON Agreement
    ("n", "t"): ("ADJ_DET_AGR", "ADJ/DET.Gender"),
    ("", "t"): ("ADJ_DET_AGR", "ADJ/DET.Neuter"),
    ("", "n"): ("ADJ_DET_AGR", "ADJ/DET.MascFem"),
    ("", "e"): ("ADJ_DET_AGR", "ADJ/NOUN.Suffix.-e"),
}


CORE_MORPHEME_ROOTS: set[tuple[str, ...]] = {
    ("ing", "ning"),
    ("else", "ing"),
    ("laus", "løs"),
    ("veg", "vei"),
    ("vann", "vatn"),
    ("gard", "gård"),
    ("hand", "hånd"),
    ("band", "bånd"),
    ("sjuk", "syk"),
    ("fram", "frem"),
    ("høg", "høy"),
    ("spell", "spill"),
    ("skott", "skudd"),
    ("brott", "brudd"),
    ("blaut", "bløt"),
    ("raud", "rød"),
    ("rein", "ren"),
    ("stein", "sten"),
    ("bein", "ben"),
    ("grein", "gren"),
    ("lauv", "løv"),
    ("haust", "høst"),
    ("daud", "død"),
    ("mjølk", "melk"),
    ("mjøl", "mel"),
    ("kol", "kull"),
    ("hol", "hull"),
    ("botn", "bunn"),
    ("auke", "øke"),
    ("fyrst", "først"),
    ("sundag", "søndag"),
    ("sein", "sen"),
    ("kjel", "kjele"),
    ("celle", "selle"),
    ("korg", "kurv"),
    ("ide", "idé"),
    ("koble", "kople"),
    ("kobl", "kopl"),
    ("megle", "mekle"),
    ("megl", "mekl"),
    ("golv", "gulv"),
    ("lign", "likn"),
    ("tredel", "tredjedel"),
    ("firedel", "fjerdedel"),
    ("ven", "venn"),
    ("koma", "komma"),
    ("kome", "komme"),
    ("kjøken", "kjøkken"),
    ("gamal", "gammal"),
    ("barn", "born"),
    ("klar", "klår"),
    ("fylg", "følg"),
    ("ynsk", "ønsk"),
    ("bro", "bru"),
    ("sju", "syv"),
    ("tjue", "tyve"),
    ("tredve", "tretti"),
    ("førti", "førr"),
    ("da", "då"),
    ("ga", "gav"),
    ("sto", "stod"),
    ("no", "nå"),
    ("me", "vi"),
    ("gir", "gjev"),
    ("tar", "tek"),
    ("tatt", "teke"),
    ("blei", "vart"),
    ("derfor", "difor"),
    ("utafor", "utenfor"),
    ("innafor", "innenfor"),
    ("fra", "ifra"),
    ("gjennom", "igjennom"),
    ("medlem", "medlemm"),
}

CANONICAL_STEM_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("kobl", "kopl"): ("koble", "kople"),
    ("megl", "mekl"): ("megle", "mekle"),
    ("meg", "mek"): ("megle", "mekle"),
    ("fyl", "føl"): ("fylg", "følg"),
    ("kom", "komm"): ("kome", "komme"),
    ("medlem", "medlemm"): ("medlem", "medlemm"),
}


def get_inflection_map(lang: str) -> dict[tuple[str, ...], tuple[str, str]]:
    return NNO_INFLECTION_MAP if lang == "nno" else NOB_INFLECTION_MAP


def _longest_common_prefix_len(words: Sequence[str]) -> int:
    p = 0
    for chars in zip(*words):
        if len(set(chars)) == 1:
            p += 1
        else:
            break
    return p


def _longest_common_suffix_len(words: Sequence[str], prefix_len: int) -> int:
    rem = [w[prefix_len:] for w in words]
    s = 0
    for chars in zip(*(r[::-1] for r in rem)):
        if len(set(chars)) == 1:
            s += 1
        else:
            break
    return s


def factor_combined_inflection(
    cands: list[str], imap: dict[tuple[str, ...], tuple[str, str]]
) -> tuple[list[str], str | None]:
    """Factor out an inflectional suffix alternation when combined with a stem alternation."""
    for sufs, (_cat, tag) in sorted(
        imap.items(), key=lambda kv: -max(len(x) for x in kv[0])
    ):
        if len(sufs) < 2:
            continue
        matched_stems: set[str] = set()
        used_sufs: set[str] = set()
        ok = True
        for c in cands:
            matched = False
            for suf in sorted(sufs, key=len, reverse=True):
                if c.endswith(suf) and len(c) - len(suf) >= 2:
                    matched_stems.add(c[: len(c) - len(suf)])
                    used_sufs.add(suf)
                    matched = True
                    break
            if not matched:
                ok = False
                break
        if (
            ok
            and len(used_sufs) == len(sufs)
            and 2 <= len(matched_stems) < len(cands)
        ):
            return sorted(matched_stems), tag

    if len(cands) == 2:
        for sufs, (_cat, tag) in imap.items():
            if len(sufs) == 2:
                for s0, s1 in (sufs, (sufs[1], sufs[0])):
                    if cands[0].endswith(s0) and cands[1].endswith(s1):
                        st0 = cands[0][: len(cands[0]) - len(s0)]
                        st1 = cands[1][: len(cands[1]) - len(s1)]
                        if (
                            len(st0) >= 3
                            and len(st1) >= 3
                            and st0 != st1
                            and st0[-2:] == st1[-2:]
                        ):
                            return sorted([st0, st1]), tag
    return cands, None


def build_minimal_roots(
    all_candidate_groups: Sequence[Sequence[str]], lang: str = "nob"
) -> set[tuple[str, ...]]:
    """Build a minimal set of simplex root pairs from a collection of candidate groups."""
    imap = get_inflection_map(lang)
    raw_roots: set[tuple[str, ...]] = set(CORE_MORPHEME_ROOTS)

    for grp in all_candidate_groups:
        cands = sorted({c.strip().lower() for c in grp if c.strip()})
        if len(cands) < 2:
            continue
        p = _longest_common_prefix_len(cands)
        sufs = tuple(c[p:] for c in cands)
        if p >= 2 and sufs in imap:
            continue
        if len(cands) == 2 and all(3 <= len(c) <= 6 for c in cands):
            raw_roots.add(tuple(cands))

    minimal_roots: set[tuple[str, ...]] = set()
    for r in sorted(raw_roots, key=lambda x: sum(len(w) for w in x)):
        if len(r) != 2:
            minimal_roots.add(r)
            continue
        r0, r1 = r
        is_extension = False
        for m in minimal_roots:
            if len(m) != 2:
                continue
            m0, m1 = m
            if len(r0) > len(m0) and len(r1) > len(m1):
                idx0 = r0.find(m0)
                idx1 = r1.find(m1)
                if idx0 != -1 and idx0 == idx1:
                    if r0[idx0 + len(m0) :] == r1[idx1 + len(m1) :]:
                        is_extension = True
                        break
        if not is_extension:
            minimal_roots.add(r)
    return minimal_roots


def classify_candidate_group(
    candidates: Sequence[str],
    *,
    lang: str = "nob",
    minimal_roots: set[tuple[str, ...]] | None = None,
) -> CategoryClassification:
    """Classify a candidate group into either a pure grammatical inflection category or a stem family."""
    cands = sorted({c.strip().lower() for c in candidates if c.strip()})
    imap = get_inflection_map(lang)
    if len(cands) < 2:
        order, title = CATEGORY_META["LEXICAL_STEM"]
        return CategoryClassification(
            category_id="LEXICAL_STEM",
            category_title=title,
            category_order=order,
            tag="STEM",
            stem_family=tuple(cands),
            is_simplex_root=True,
        )

    p = _longest_common_prefix_len(cands)
    sufs = tuple(c[p:] for c in cands)
    if p >= 2 and sufs in imap:
        cat_id, tag = imap[sufs]
        order, title = CATEGORY_META[cat_id]
        return CategoryClassification(
            category_id=cat_id,
            category_title=title,
            category_order=order,
            tag=tag,
            stem_family=None,
            is_simplex_root=False,
        )

    roots = minimal_roots if minimal_roots is not None else CORE_MORPHEME_ROOTS
    factored_cands, inf_tag = factor_combined_inflection(cands, imap)
    fp = _longest_common_prefix_len(factored_cands)
    fs = _longest_common_suffix_len(factored_cands, fp)

    matched_family: tuple[str, ...] | None = None
    for fwd in range(0, min(fs, 7) + 1):
        for back in range(0, min(fp, 7) + 1):
            start = fp - back
            end_trim = fs - fwd
            cores = tuple(
                sorted(
                    {
                        c[start : len(c) - end_trim if end_trim > 0 else len(c)]
                        for c in factored_cands
                    }
                )
            )
            if len(cores) >= 2 and cores in roots:
                matched_family = CANONICAL_STEM_ALIASES.get(cores, cores)
                break
        if matched_family is not None:
            break

    if matched_family is None:
        if all(len(c) <= 8 for c in factored_cands):
            matched_family = tuple(factored_cands)
        else:
            start = max(0, fp - 2) if fp > 0 else 0
            end_trim = max(0, fs - 2) if fs > 2 else 0
            matched_family = tuple(
                sorted(
                    {
                        c[start : len(c) - end_trim if end_trim > 0 else len(c)]
                        for c in factored_cands
                    }
                )
            )
        matched_family = CANONICAL_STEM_ALIASES.get(matched_family, matched_family)

    stem_str = "/".join(matched_family)
    tag = f"STEM:{stem_str} + {inf_tag}" if inf_tag else f"STEM:{stem_str}"
    order, title = CATEGORY_META["LEXICAL_STEM"]
    is_simplex = tuple(cands) == matched_family
    return CategoryClassification(
        category_id="LEXICAL_STEM",
        category_title=title,
        category_order=order,
        tag=tag,
        stem_family=matched_family,
        is_simplex_root=is_simplex,
    )


def organize_choices_by_category(
    entries: Sequence[dict[str, Any]],
    *,
    lang: str = "nob",
) -> list[dict[str, Any]]:
    """Group and sort choice entries by grammatical category (and stem family within LEXICAL_STEM).

    Each input dict in `entries` must contain:
      - 'cands': Sequence[str]
      - 'disp': str (display key, e.g. 'boka | boken')
      - 'chosen': str
      - 'seen': int
      - 'changed': int
      - 'stats_comment': str (e.g. 'seen: 10, changed: 5, ...' without leading '#')
    """
    minimal_roots = build_minimal_roots([e["cands"] for e in entries], lang=lang)
    by_category: dict[str, list[tuple[CategoryClassification, dict[str, Any]]]] = (
        defaultdict(list)
    )

    for entry in entries:
        cls = classify_candidate_group(
            entry["cands"], lang=lang, minimal_roots=minimal_roots
        )
        by_category[cls.category_id].append((cls, entry))

    sections: list[dict[str, Any]] = []
    for cat_id, (cat_order, cat_title) in sorted(
        CATEGORY_META.items(), key=lambda kv: kv[1][0]
    ):
        cat_items = by_category.get(cat_id, [])
        if not cat_items:
            continue

        if cat_id != "LEXICAL_STEM":
            cat_items.sort(
                key=lambda item: (
                    -int(item[1].get("changed", 0)),
                    -int(item[1].get("seen", 0)),
                    str(item[1]["disp"]).casefold(),
                )
            )
            formatted_rows = []
            for cls, entry in cat_items:
                stats_str = entry.get("stats_comment", "").strip()
                comment = (
                    f"  # [{cls.tag}] {stats_str}"
                    if stats_str
                    else f"  # [{cls.tag}]"
                )
                formatted_rows.append(
                    {
                        "disp": entry["disp"],
                        "chosen": entry["chosen"],
                        "comment": comment,
                        "tag": cls.tag,
                        "seen": int(entry.get("seen", 0)),
                        "changed": int(entry.get("changed", 0)),
                        "subheader": None,
                    }
                )
            sections.append(
                {
                    "category_id": cat_id,
                    "category_order": cat_order,
                    "category_title": cat_title,
                    "total_entries": len(formatted_rows),
                    "total_changed": sum(r["changed"] for r in formatted_rows),
                    "total_seen": sum(r["seen"] for r in formatted_rows),
                    "rows": formatted_rows,
                }
            )
        else:
            # Group by stem_family, sort families by (total_changed desc, total_seen desc, len desc, family_key)
            families: dict[
                tuple[str, ...], list[tuple[CategoryClassification, dict[str, Any]]]
            ] = defaultdict(list)
            for cls, entry in cat_items:
                fam = cls.stem_family or tuple(entry["cands"])
                families[fam].append((cls, entry))

            ranked_families = sorted(
                families.items(),
                key=lambda kv: (
                    -sum(int(x[1].get("changed", 0)) for x in kv[1]),
                    -sum(int(x[1].get("seen", 0)) for x in kv[1]),
                    -len(kv[1]),
                    kv[0],
                ),
            )
            formatted_rows = []
            for fam, fam_items in ranked_families:
                fam_items.sort(
                    key=lambda item: (
                        not item[0].is_simplex_root,
                        -int(item[1].get("changed", 0)),
                        -int(item[1].get("seen", 0)),
                        str(item[1]["disp"]).casefold(),
                    )
                )
                fam_label = " / ".join(fam)
                fam_chg = sum(int(x[1].get("changed", 0)) for x in fam_items)
                fam_seen = sum(int(x[1].get("seen", 0)) for x in fam_items)
                subheader = (
                    f"--- Stem family: {fam_label} "
                    f"({len(fam_items)} entries, changed: {fam_chg:,}, seen: {fam_seen:,}) ---"
                    if len(fam_items) > 1 or fam_chg > 0
                    else None
                )
                for idx, (cls, entry) in enumerate(fam_items):
                    stats_str = entry.get("stats_comment", "").strip()
                    comment = (
                        f"  # [{cls.tag}] {stats_str}"
                        if stats_str
                        else f"  # [{cls.tag}]"
                    )
                    formatted_rows.append(
                        {
                            "disp": entry["disp"],
                            "chosen": entry["chosen"],
                            "comment": comment,
                            "tag": cls.tag,
                            "seen": int(entry.get("seen", 0)),
                            "changed": int(entry.get("changed", 0)),
                            "subheader": subheader if idx == 0 else None,
                        }
                    )
            sections.append(
                {
                    "category_id": cat_id,
                    "category_order": cat_order,
                    "category_title": cat_title,
                    "total_entries": len(formatted_rows),
                    "total_changed": sum(r["changed"] for r in formatted_rows),
                    "total_seen": sum(r["seen"] for r in formatted_rows),
                    "rows": formatted_rows,
                }
            )

    return sections

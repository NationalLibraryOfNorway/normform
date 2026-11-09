"""Empirical style evidence selection based on model likelihood gains."""
from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any, Sequence

from .contract import STYLE_RULE, STYLE_TASK, parse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else []


@dataclass(frozen=True)
class DocumentGroup:
    sentence_index: int
    group_index: int
    surface: str
    candidates: tuple[str, ...]
    sentence_text: str


@dataclass(frozen=True)
class SelectionResult:
    words: list[str]
    explained_variance: float
    total_variance: float
    gains: dict[str, float]
    details: list[dict[str, Any]]


def extract_document_groups(
    adapter: Any,
    sentences: Sequence[str],
    *,
    show_progress: bool = False,
) -> list[DocumentGroup]:
    groups: list[DocumentGroup] = []
    iterator = (
        tqdm(
            sentences,
            desc="[extract_style] Generating sentence alternatives",
            file=sys.stderr,
        )
        if show_progress
        else sentences
    )
    for s_idx, text in enumerate(iterator):
        stripped = text.strip()
        if not stripped:
            continue
        try:
            alt_res = adapter.alternatives(stripped)
            annotated = alt_res.get("text", stripped)
            template = parse(annotated)
            if not template.groups:
                continue
            aligned = template.align(stripped)
            for g_idx, candidates in enumerate(template.groups):
                surface = aligned[g_idx].lower()
                groups.append(
                    DocumentGroup(
                        sentence_index=s_idx,
                        group_index=g_idx,
                        surface=surface,
                        candidates=candidates,
                        sentence_text=annotated,
                    )
                )
        except Exception:
            continue
    return groups


def _score_sentences(
    adapter: Any,
    sentence_items: list[tuple[int, str]],
    evidence: Sequence[str],
) -> dict[int, float]:
    if not sentence_items:
        return {}

    evidence_str = " | ".join(evidence)
    results: dict[int, float] = {}

    prompts = [
        f"{STYLE_TASK}:\n{STYLE_RULE}\n"
        f"morfologiske_stilvalg: {evidence_str}\n"
        f"setning: {text}"
        for _, text in sentence_items
    ]
    targets = [text for _, text in sentence_items]

    if hasattr(adapter, "score_pairs"):
        try:
            scores = adapter.score_pairs(prompts, targets)
            for (s_idx, _), val in zip(sentence_items, scores):
                results[s_idx] = float(val)
            return results
        except Exception:
            pass

    for (s_idx, input_text), prompt in zip(sentence_items, prompts):
        try:
            scores = adapter.score(prompt, [input_text])
            results[s_idx] = float(scores[0])
        except Exception:
            results[s_idx] = 0.0
    return results


def find_representative_evidence(
    adapter: Any,
    document: str | Sequence[str],
    n: int = 10,
    *,
    lang: str = "nob",
    candidate_pool_size: int = 30,
    show_progress: bool = True,
) -> SelectionResult:
    """Extract N most representative evidence prompt words by empirical likelihood gain."""
    if isinstance(document, str):
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", document) if s.strip()
        ]
    else:
        raw_sentences = [str(s).strip() for s in document if str(s).strip()]

    doc_groups = extract_document_groups(
        adapter, raw_sentences, show_progress=show_progress
    )
    if not doc_groups:
        return SelectionResult(
            words=[],
            explained_variance=0.0,
            total_variance=0.0,
            gains={},
            details=[],
        )

    unique_choices: list[str] = []
    seen_choices: set[str] = set()
    for g in doc_groups:
        lowercased = g.surface.lower()
        if lowercased not in seen_choices:
            seen_choices.add(lowercased)
            unique_choices.append(lowercased)

    sentence_map: dict[int, str] = {}
    for g in doc_groups:
        sentence_map[g.sentence_index] = g.sentence_text
    sentence_items = sorted(sentence_map.items())

    if show_progress:
        print(
            f"[extract_style] Analyzing {len(sentence_items)} sentences with {len(unique_choices)} candidate style choices...",
            file=sys.stderr,
            flush=True,
        )

    baseline_scores = _score_sentences(adapter, sentence_items, [])
    baseline_sum = sum(baseline_scores.values())

    full_scores = _score_sentences(adapter, sentence_items, unique_choices)
    full_sum = sum(full_scores.values())
    total_variance = max(0.0, full_sum - baseline_sum)

    single_gains: dict[str, float] = {}
    phase2_iter = (
        tqdm(
            unique_choices,
            desc="[extract_style] Single-word impact evaluation",
            file=sys.stderr,
        )
        if show_progress
        else unique_choices
    )
    for word in phase2_iter:
        word_scores = _score_sentences(adapter, sentence_items, [word])
        gain = sum(word_scores.values()) - baseline_sum
        single_gains[word] = gain

    sorted_candidates = sorted(
        unique_choices, key=lambda w: single_gains[w], reverse=True
    )
    candidate_pool = sorted_candidates[: min(candidate_pool_size, len(sorted_candidates))]

    selected: list[str] = []
    current_pool = list(candidate_pool)
    step_details: list[dict[str, Any]] = []

    current_best_score = baseline_sum
    target_n = min(n, len(candidate_pool))

    phase3_iter = (
        tqdm(
            range(target_n),
            desc="[extract_style] Joint forward selection",
            file=sys.stderr,
        )
        if show_progress
        else range(target_n)
    )
    for step in phase3_iter:
        best_word = None
        best_gain = -1e9

        for candidate in current_pool:
            test_evidence = selected + [candidate]
            scores = _score_sentences(adapter, sentence_items, test_evidence)
            score_sum = sum(scores.values())
            if score_sum > best_gain:
                best_gain = score_sum
                best_word = candidate

        if best_word is None:
            break

        selected.append(best_word)
        current_pool.remove(best_word)

        incremental_gain = best_gain - current_best_score
        current_best_score = best_gain

        step_details.append(
            {
                "step": step + 1,
                "word": best_word,
                "total_score": best_gain,
                "incremental_gain": incremental_gain,
                "single_gain": single_gains[best_word],
            }
        )

    if total_variance > 1e-6:
        explained_ratio = (current_best_score - baseline_sum) / total_variance
        explained_variance = max(0.0, min(100.0, float(explained_ratio * 100.0)))
    else:
        explained_variance = 100.0 if selected else 0.0

    return SelectionResult(
        words=selected,
        explained_variance=explained_variance,
        total_variance=total_variance,
        gains=single_gains,
        details=step_details,
    )

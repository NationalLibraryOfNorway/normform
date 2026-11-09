"""High-throughput 1-GPU streaming corpus standardization engine."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Sequence

import yaml

from .audit import extract_changed_alternatives
from .categories import organize_choices_by_category
from .contract import Template, clean_direct, parse
from .harness import parse_prompt_words, split_sentences
from .model import DEFAULT_MODEL_BY_LANG, load_adapter, model_revision
from .styles import (
    canonical_choice_key,
    invalid_pair_signature,
    is_invalid_pair,
    load_invalid_pairs,
    load_seed_cache,
    load_yaml,
    resolve_candidate_choice,
    resolve_style_path,
    yaml_choice_scalar,
)


def estimate_record_count(path: Path) -> int | None:
    """Quickly estimate or count records in the input file or directory."""
    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            try:
                import pyarrow.parquet as pq

                return sum(pq.read_metadata(p).num_rows for p in parquet_files)
            except Exception:
                return None
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            try:
                total = 0
                for jf in jsonl_files:
                    res = subprocess.run(
                        ["wc", "-l", str(jf)],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    total += int(res.stdout.strip().split()[0])
                return total
            except Exception:
                return None
        return None
    if not path.is_file():
        return None
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return pq.read_metadata(path).num_rows
        except Exception:
            return None
    try:
        res = subprocess.run(
            ["wc", "-l", str(path)], capture_output=True, text=True, check=True
        )
        return int(res.stdout.strip().split()[0])
    except Exception:
        return None


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable minutes and hours."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"~{int(minutes)} minutes ({int(seconds)}s)"
    hours = minutes / 60
    rem_min = int(minutes % 60)
    return f"~{hours:.1f} hours ({int(hours)}h {rem_min}m)"


def is_huggingface_dataset(input_str: str) -> bool:
    """Check if input string refers to a Hugging Face dataset."""
    path = Path(input_str)
    if path.exists():
        return False
    if input_str.startswith("hf://"):
        return True
    if "/" in input_str and not input_str.startswith((".", "/")):
        return True
    return False


def read_records(
    source: str | Path,
    text_field: str,
    limit: int | None = None,
    dataset_revision: str | None = None,
    dataset_split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Stream records from JSONL, Parquet, TXT, directory shards, or Hugging Face dataset."""
    src_str = str(source)
    if is_huggingface_dataset(src_str):
        import datasets

        clean_id = src_str.removeprefix("hf://")
        ds = datasets.load_dataset(
            clean_id, split=dataset_split, streaming=True, revision=dataset_revision
        )
        count = 0
        for row in ds:
            yield dict(row)
            count += 1
            if limit and count >= limit:
                break
        return

    path = Path(source)
    count = 0
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            import pyarrow.parquet as pq

            for pfile in parquet_files:
                table = pq.read_table(pfile)
                for row in table.to_pylist():
                    yield row
                    count += 1
                    if limit and count >= limit:
                        return
            return
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            for jfile in jsonl_files:
                with jfile.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
                            count += 1
                            if limit and count >= limit:
                                return
            return
        raise ValueError(f"Directory {path} contains no .parquet or .jsonl shards.")

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row
            count += 1
            if limit and count >= limit:
                break
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if limit and count >= limit:
                        break
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield {text_field: stripped}
                    count += 1
                    if limit and count >= limit:
                        break


def greedy_set_cover(
    candidate_to_sentences: dict[str, list[int]],
    sentence_to_candidates: dict[int, list[str]],
) -> list[int]:
    """Greedy set cover: pick minimum subset of sentences covering all unique missing candidate groups."""
    uncovered = set(candidate_to_sentences.keys())
    selected_sentences: list[int] = []
    sent_map = {s_idx: set(cands) for s_idx, cands in sentence_to_candidates.items()}

    while uncovered:
        best_sent = -1
        best_cover_count = 0
        for s_idx, cands in sent_map.items():
            overlap = len(cands & uncovered)
            if overlap > best_cover_count:
                best_cover_count = overlap
                best_sent = s_idx

        if best_sent == -1 or best_cover_count == 0:
            break

        selected_sentences.append(best_sent)
        uncovered -= sent_map[best_sent]
        del sent_map[best_sent]

    return selected_sentences


def load_prompt_words(
    prompt_arg: str | None, seed_cache_path: Path | None = None
) -> list[str]:
    """Load explicit prompt words, or reuse those embedded in the style cache."""
    if not prompt_arg:
        if seed_cache_path is None or not seed_cache_path.is_file():
            return []
        cache = load_yaml(seed_cache_path.read_text(encoding="utf-8"))
        words = (
            cache.get("_metadata", {}).get("prompt_words", [])
            if isinstance(cache, dict)
            else []
        )
        return parse_prompt_words(words)
    path = Path(prompt_arg)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            data = json.loads(text)
            return list(data.get("prompt_words", []))
        return parse_prompt_words(text)
    return parse_prompt_words(prompt_arg)


def export_sealed_cache(
    choices: dict[str, str],
    output_path: Path,
    metadata: dict[str, Any],
    corpus_path: Path | None = None,
    text_field: str = "text",
    normalized_field: str = "normalized",
    lang: str = "nob",
    decision_stats: dict[str, dict[str, Any]] | None = None,
    invalid_pairs: dict[tuple[str, ...], str] | None = None,
) -> None:
    """Save completed sealed cache in human-readable YAML top-sorted by corpus change impact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_choices: dict[tuple[str, ...], str] = {}
    key_to_display: dict[tuple[str, ...], str] = {}
    canonical_decision_stats: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "seen": 0,
            "changed": 0,
            "already_correct": 0,
            "changed_inputs": Counter(),
        }
    )

    for key, chosen in choices.items():
        if "\x1f" in key:
            raw_cands = [c.strip() for c in key.split("\x1f")]
        elif " | " in key:
            raw_cands = [c.strip() for c in key.split(" | ")]
        else:
            raw_cands = [c.strip() for c in key.split(",")]
        cands = tuple(sorted({c.lower() for c in raw_cands}))
        normalized_choice = chosen.lower()
        previous = canonical_choices.get(cands)
        if previous is not None and previous != normalized_choice:
            raise ValueError(f"Conflicting case variants for cache group {cands!r}")
        canonical_choices[cands] = normalized_choice
        key_to_display[cands] = " | ".join(cands)

        if decision_stats is not None and key in decision_stats:
            source = decision_stats[key]
            target = canonical_decision_stats[cands]
            target["seen"] += int(source.get("seen", 0))
            target["changed"] += int(source.get("changed", 0))
            target["already_correct"] += int(source.get("already_correct", 0))
            target["changed_inputs"].update(source.get("changed_inputs", {}))

    group_change_stats: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    group_total_changes: dict[tuple[str, ...], int] = {}
    total_scanned = 0
    total_changed = 0

    if corpus_path and corpus_path.exists():
        try:
            ranked, total_scanned, total_changed = extract_changed_alternatives(
                corpus_path=corpus_path,
                text_field=text_field,
                normalized_field=normalized_field,
                cache_path=None,
                lang=lang,
            )
            for grp, _, tot, details in ranked:
                cands = tuple(sorted({c.lower() for c in grp}))
                group_total_changes[cands] = tot
                group_change_stats[cands] = details
        except Exception as exc:
            print(
                f"  ⚠️ Warning: Could not extract change stats for cache export: {exc}",
                file=sys.stderr,
            )

    raw_entries: list[dict[str, Any]] = []
    changed_count = 0
    unchanged_count = 0

    for cands, chosen in canonical_choices.items():
        disp = key_to_display.get(cands, " | ".join(cands))
        exact = canonical_decision_stats[cands] if decision_stats is not None else None
        if exact is not None:
            seen = int(exact["seen"])
            tot_chg = int(exact["changed"])
            already_correct = int(exact["already_correct"])
            details = Counter(exact["changed_inputs"]).most_common()
            change_rate = 100.0 * tot_chg / seen if seen else 0.0
            detail_strs = [f"{word}: {count}" for word, count in details if count > 0]
            detail_part = f" ({', '.join(detail_strs)})" if detail_strs else ""
            stats_comment = (
                f"seen: {seen}, changed: {tot_chg}{detail_part}, "
                f"already_correct: {already_correct}, change_rate: {change_rate:.2f}%"
            )
        else:
            tot_chg = group_total_changes.get(cands, 0)
            seen = tot_chg
            details = group_change_stats.get(cands, [])
            detail_strs = [f"{w}: {c}" for w, c in details if c > 0]
            if tot_chg > 0:
                stats_comment = (
                    f"changed: {tot_chg} ({', '.join(detail_strs)})"
                    if detail_strs
                    else f"changed: {tot_chg}"
                )
            else:
                stats_comment = ""

        if tot_chg > 0:
            changed_count += 1
        else:
            unchanged_count += 1

        raw_entries.append(
            {
                "cands": cands,
                "disp": disp,
                "chosen": chosen,
                "seen": seen,
                "changed": tot_chg,
                "stats_comment": stats_comment,
            }
        )

    sections = organize_choices_by_category(raw_entries, lang=lang)

    pct_changed = (
        (total_changed / max(1, total_scanned)) * 100.0 if total_scanned else 0.0
    )
    updated_meta = dict(metadata)
    if total_scanned:
        updated_meta["total_records_scanned"] = f"{total_scanned:,}"
        updated_meta["total_records_changed"] = (
            f"{total_changed:,} ({pct_changed:.2f}%)"
        )
    updated_meta["changed_choices_count"] = changed_count
    updated_meta["unchanged_choices_count"] = unchanged_count
    updated_meta["sort_order"] = "grammatical_category_and_stem_family_then_frequency"
    if invalid_pairs:
        updated_meta["invalid_pairs_count"] = len(invalid_pairs)
    if decision_stats is not None:
        total_seen = sum(
            int(item["seen"]) for item in canonical_decision_stats.values()
        )
        total_decisions_changed = sum(
            int(item["changed"]) for item in canonical_decision_stats.values()
        )
        total_already_correct = sum(
            int(item["already_correct"]) for item in canonical_decision_stats.values()
        )
        updated_meta["decision_statistics"] = {
            "method": "exact_lattice_events",
            "seen": total_seen,
            "changed": total_decisions_changed,
            "already_correct": total_already_correct,
            "change_rate_percent": round(
                100.0 * total_decisions_changed / total_seen if total_seen else 0.0,
                4,
            ),
        }

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# ====================================================================\n")
        f.write("# NormForm Completed & Sealed Style Cache (Unified & Human-Editable)\n")
        f.write(f"# Language: {metadata.get('language', lang)}\n")
        f.write(f"# Sealed at: {metadata.get('sealed_at', '')}\n")
        f.write(
            f"# Total Choices: {len(canonical_choices):,} "
            f"({changed_count:,} changed in corpus, {unchanged_count:,} reference)\n"
        )
        f.write("# Organization: Grammatical Category & Stem Family -> Frequency (changed, seen)\n")
        f.write("# ====================================================================\n\n")

        f.write(
            yaml.safe_dump(
                {"_metadata": updated_meta}, allow_unicode=True, sort_keys=False
            ).strip()
            + "\n\nchoices:\n"
        )

        for section in sections:
            f.write(
                f"\n  # =========================================================================\n"
                f"  # {section['category_order']}. {section['category_title']}\n"
                f"  #    Entries: {section['total_entries']:,} | Changed: {section['total_changed']:,} | Seen: {section['total_seen']:,}\n"
                f"  # =========================================================================\n"
            )
            for row in section["rows"]:
                if row.get("subheader"):
                    f.write(f"\n  # {row['subheader']}\n")
                f.write(
                    f"  {yaml_choice_scalar(row['disp'])}: {yaml_choice_scalar(row['chosen'])}{row['comment']}\n"
                )

        if invalid_pairs:
            display = {
                " | ".join(invalid_pair_signature(key)): reason
                for key, reason in sorted(invalid_pairs.items())
            }
            f.write(
                "\n"
                + yaml.safe_dump(
                    {"invalid_pairs": display}, allow_unicode=True, sort_keys=False
                )
            )


def record_choice_decision(
    decision_stats: dict[str, dict[str, Any]],
    key: str,
    original_choice: str,
    cached_choice: str,
) -> None:
    """Record one exact cache decision made for one emitted lattice group."""
    stats = decision_stats.setdefault(
        key,
        {
            "seen": 0,
            "changed": 0,
            "already_correct": 0,
            "changed_inputs": Counter(),
        },
    )
    stats["seen"] += 1
    if original_choice == cached_choice:
        stats["already_correct"] += 1
    else:
        stats["changed"] += 1
        stats["changed_inputs"][original_choice] += 1


def render_cached_template(
    template: Template,
    original_text: str,
    cache: dict[str, str],
    invalid_pairs: dict[tuple[str, ...], str],
    decision_stats: dict[str, dict[str, Any]],
    *,
    original_choices: tuple[str, ...] | None = None,
) -> str:
    """Render legal candidate spellings, then record successful cache hits."""
    if original_choices is None:
        original_choices = template.align(original_text)
    chosen_list = []
    applied_decisions = []
    for candidates, original_choice in zip(template.groups, original_choices):
        key = canonical_choice_key(candidates)
        chosen, cache_hit = resolve_candidate_choice(
            candidates, original_choice, cache.get(key), invalid_pairs
        )
        chosen_list.append(chosen)
        if cache_hit:
            applied_decisions.append((key, original_choice, chosen))
    rendered = template.render(tuple(chosen_list))
    for key, original_choice, chosen in applied_decisions:
        record_choice_decision(decision_stats, key, original_choice, chosen)
    return rendered


def alternatives_batch_resilient(
    adapter: Any,
    texts: Sequence[str],
) -> tuple[list[str], int]:
    """Expand a batch while preserving isolated over-capacity texts unchanged."""
    try:
        return list(adapter.alternatives_batch(texts)), 0
    except ValueError as exc:
        if "Prompt exceeds model input capacity" not in str(exc):
            raise
        if len(texts) == 1:
            return [texts[0]], 1

        midpoint = len(texts) // 2
        left, left_skipped = alternatives_batch_resilient(adapter, texts[:midpoint])
        right, right_skipped = alternatives_batch_resilient(adapter, texts[midpoint:])
        return left + right, left_skipped + right_skipped


def run_batch_standardization(
    input_source: str | Path,
    output_path: Path,
    sealed_path: Path,
    *,
    lang: str = "nob",
    style: str | Path | None = None,
    seed_cache_path: Path | None = None,
    prompt_words_arg: str | None = None,
    preserve_unreviewed: bool = False,
    text_field: str = "text",
    output_field: str = "normalized",
    dataset_split: str = "train",
    dataset_revision: str | None = None,
    batch_size: int = 512,
    chunk_size: int = 4096,
    limit: int | None = None,
    model_id: str | None = None,
    config_path: Path | None = None,
    device: str = "auto",
) -> int:
    """Execute the 3-phase high-throughput standardization directly on GPU."""
    start_time = time.time()
    resolved_cache_path = seed_cache_path or resolve_style_path(style, lang=lang)
    if preserve_unreviewed and resolved_cache_path is None:
        raise ValueError("--preserve-unreviewed requires --style or --seed-cache")

    print("=" * 76)
    print("  🚀 NormForm High-Throughput 1-GPU Standardization Engine")
    print("=" * 76)
    print(f"  Input Corpus    : {input_source}")
    print(f"  Output File     : {output_path}")
    print(f"  Language        : {lang}")
    print(f"  Batch Size      : {batch_size}")
    print(f"  Device          : {device}")

    invalid_pairs = load_invalid_pairs(resolved_cache_path)
    cache = {
        key: choice
        for key, choice in load_seed_cache(resolved_cache_path).items()
        if not is_invalid_pair(key.split("\x1f"), invalid_pairs)
    }
    prompt_words = load_prompt_words(prompt_words_arg, resolved_cache_path)
    print(f"  Style Cache     : {len(cache)} pre-existing choices loaded")
    print(f"  Invalid Pairs   : {len(invalid_pairs)} exact candidate sets preserved unchanged")
    print(
        f"  Unreviewed Sets : {'preserve source form' if preserve_unreviewed else 'model style selection'}"
    )
    print(
        f"  Prompt Words    : {', '.join(prompt_words) if prompt_words else '(none provided)'}"
    )

    print(f"\n[Phase 0/3] 📦 Loading model on {device}...", flush=True)
    effective_model_id = model_id or DEFAULT_MODEL_BY_LANG[lang]
    rev = model_revision(effective_model_id)
    print(f"  Model           : {effective_model_id}@{rev or 'local/unpinned'}", flush=True)
    adapter, _ = load_adapter(
        config_path,
        effective_model_id,
        device_name=device,
        require_multitask=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_parquet = output_path.suffix == ".parquet"
    is_txt = output_path.suffix == ".txt" and str(input_source).endswith(".txt")

    estimated_total = (
        estimate_record_count(Path(str(input_source)))
        if not is_huggingface_dataset(str(input_source))
        else None
    )
    if estimated_total is None and limit:
        estimated_total = limit

    print(
        f"\n[Phase 1/2] ⚡ Streaming Batched Normalization on GPU ({device})...",
        flush=True,
    )
    stream_start = time.time()
    total_records = 0
    total_sentences = 0
    valid_lattices = 0
    new_choices_added = 0
    overlong_sentences_skipped = 0
    decision_stats: dict[str, dict[str, Any]] = {}
    parquet_writer = None

    def prefetch_and_presplit_chunks(
        iterable: Iterator[dict[str, Any]],
        c_size: int,
        t_field: str,
        queue_size: int = 2,
    ) -> Iterator[tuple[list[dict[str, Any]], list[tuple[int, str, str, str]]]]:
        chunk_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        sentinel = object()
        err_holder: list[Exception] = []

        def producer():
            try:
                curr_records: list[dict[str, Any]] = []
                curr_items: list[tuple[int, str, str, str]] = []

                for rec in iterable:
                    r_idx = len(curr_records)
                    curr_records.append(rec)
                    if t_field not in rec:
                        raise ValueError(
                            f"Source text field {t_field!r} is missing from input record"
                        )
                    text = str(rec[t_field])
                    for span in split_sentences(text):
                        raw_txt = span.text
                        l_ws = raw_txt[: len(raw_txt) - len(raw_txt.lstrip())]
                        r_ws = raw_txt[len(raw_txt.rstrip()) :]
                        curr_items.append((r_idx, raw_txt.strip(), l_ws, r_ws))

                    if len(curr_records) >= c_size:
                        chunk_queue.put((curr_records, curr_items))
                        curr_records = []
                        curr_items = []

                if curr_records:
                    chunk_queue.put((curr_records, curr_items))
            except Exception as exc:
                err_holder.append(exc)
            finally:
                chunk_queue.put(sentinel)

        reader_thread = threading.Thread(
            target=producer, daemon=True, name="PrefetchThread"
        )
        reader_thread.start()

        while True:
            item = chunk_queue.get()
            if item is sentinel:
                if err_holder:
                    raise err_holder[0]
                break
            yield item

    out_f = None
    try:
        if not is_parquet:
            out_f = output_path.open("w", encoding="utf-8")

        write_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        write_sentinel = object()
        write_err: list[Exception] = []

        def writer_loop():
            nonlocal parquet_writer
            try:
                while True:
                    task = write_queue.get()
                    if task is write_sentinel:
                        break
                    records_to_write, rec_sentences_to_write = task
                    if is_parquet:
                        import pyarrow as pa
                        import pyarrow.parquet as pq

                        for r_idx, rec in enumerate(records_to_write):
                            rec[output_field] = "".join(rec_sentences_to_write[r_idx])
                        table = pa.Table.from_pylist(records_to_write)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(
                                output_path, table.schema, compression="zstd"
                            )
                        parquet_writer.write_table(table)
                    elif is_txt:
                        for r_idx, _ in enumerate(records_to_write):
                            assert out_f is not None
                            out_f.write("".join(rec_sentences_to_write[r_idx]) + "\n")
                    else:
                        for r_idx, rec in enumerate(records_to_write):
                            assert out_f is not None
                            rec[output_field] = "".join(rec_sentences_to_write[r_idx])
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as exc:
                write_err.append(exc)

        writer_thread = threading.Thread(
            target=writer_loop, daemon=True, name="DiskWriterThread"
        )
        writer_thread.start()

        for _chunk_idx, (records_chunk, chunk_items) in enumerate(
            prefetch_and_presplit_chunks(
                read_records(
                    input_source,
                    text_field,
                    limit=limit,
                    dataset_revision=dataset_revision,
                    dataset_split=dataset_split,
                ),
                chunk_size,
                text_field,
            )
        ):
            nonblank_indices = [i for i, item in enumerate(chunk_items) if item[1]]
            sorted_nonblank = sorted(
                nonblank_indices, key=lambda i: len(chunk_items[i][1])
            )
            num_sub_batches = (len(sorted_nonblank) + batch_size - 1) // batch_size
            expanded_map: dict[int, str] = {}

            for b_idx in range(num_sub_batches):
                sub_indices = sorted_nonblank[
                    b_idx * batch_size : (b_idx + 1) * batch_size
                ]
                sub_raw_texts = [chunk_items[i][1] for i in sub_indices]

                if hasattr(adapter, "alternatives_batch"):
                    sub_results, skipped = alternatives_batch_resilient(
                        adapter, sub_raw_texts
                    )
                    if skipped:
                        overlong_sentences_skipped += skipped
                        print(
                            f"  Warning: preserved {skipped} over-capacity sentence(s) unchanged "
                            f"({overlong_sentences_skipped} total).",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    sub_results = []
                    for raw_t in sub_raw_texts:
                        res = adapter.alternatives(raw_t)
                        sub_results.append(res.get("text", raw_t))

                for orig_idx, prop in zip(sub_indices, sub_results):
                    expanded_map[orig_idx] = prop

            # Parse once per sentence and use greedy_set_cover for unseen candidate groups
            parsed_lattices: dict[int, tuple[str, Template, tuple[str, ...]]] = {}
            candidate_to_sentences: dict[str, list[int]] = defaultdict(list)
            sentence_to_candidates: dict[int, list[str]] = defaultdict(list)

            for orig_idx in nonblank_indices:
                raw_t = chunk_items[orig_idx][1]
                prop = expanded_map.get(orig_idx, raw_t)
                if "[" in prop and "]" in prop:
                    try:
                        canonical = clean_direct(raw_t, prop)
                        tmpl = parse(canonical)
                        orig_choices = tmpl.align(raw_t)
                        parsed_lattices[orig_idx] = (canonical, tmpl, orig_choices)
                        valid_lattices += 1
                        if preserve_unreviewed:
                            continue
                        for grp in tmpl.groups:
                            k = canonical_choice_key(grp)
                            if is_invalid_pair(grp, invalid_pairs) or k in cache:
                                continue
                            candidate_to_sentences[k].append(orig_idx)
                            sentence_to_candidates[orig_idx].append(k)
                    except Exception:
                        pass

            if candidate_to_sentences and prompt_words:
                selected_sent_indices = greedy_set_cover(
                    candidate_to_sentences, sentence_to_candidates
                )
                needed_sents = [
                    parsed_lattices[s_idx][0] for s_idx in selected_sent_indices
                ]
                if needed_sents:
                    if hasattr(adapter, "style_batch"):
                        style_results = adapter.style_batch(
                            prompt_words, needed_sents, batch_size=min(256, batch_size)
                        )
                    else:
                        style_results = [
                            adapter.style(prompt_words, s) for s in needed_sents
                        ]

                    for s_idx, res in zip(selected_sent_indices, style_results):
                        canonical, tmpl, _ = parsed_lattices[s_idx]
                        try:
                            aligned = tmpl.align(res.get("text", canonical))
                            for g_idx, grp in enumerate(tmpl.groups):
                                gk = canonical_choice_key(grp)
                                if gk not in cache and not is_invalid_pair(
                                    grp, invalid_pairs
                                ):
                                    cache[gk] = aligned[g_idx].lower()
                                    new_choices_added += 1
                        except Exception:
                            for grp in tmpl.groups:
                                gk = canonical_choice_key(grp)
                                if gk not in cache and not is_invalid_pair(
                                    grp, invalid_pairs
                                ):
                                    cache[gk] = grp[0].lower()
                                    new_choices_added += 1
            elif candidate_to_sentences:
                for s_idx in sentence_to_candidates:
                    _, tmpl, _ = parsed_lattices[s_idx]
                    for grp in tmpl.groups:
                        gk = canonical_choice_key(grp)
                        if gk not in cache and not is_invalid_pair(grp, invalid_pairs):
                            cache[gk] = grp[0].lower()
                            new_choices_added += 1

            rendered_sentences: list[str] = [""] * len(chunk_items)
            for i, (_r_idx, raw_t, l_ws, r_ws) in enumerate(chunk_items):
                if not raw_t:
                    rendered_sentences[i] = f"{l_ws}{r_ws}"
                    continue
                parsed_item = parsed_lattices.get(i)
                if parsed_item is not None:
                    _canonical, tmpl, orig_choices = parsed_item
                    try:
                        final_t = render_cached_template(
                            tmpl,
                            raw_t,
                            cache,
                            invalid_pairs,
                            decision_stats,
                            original_choices=orig_choices,
                        )
                        rendered_sentences[i] = f"{l_ws}{final_t}{r_ws}"
                    except Exception:
                        rendered_sentences[i] = f"{l_ws}{raw_t}{r_ws}"
                else:
                    rendered_sentences[i] = f"{l_ws}{raw_t}{r_ws}"

            rec_sentences: dict[int, list[str]] = defaultdict(list)
            for i, (r_idx, _, _, _) in enumerate(chunk_items):
                rec_sentences[r_idx].append(rendered_sentences[i])

            write_queue.put((records_chunk, rec_sentences))

            total_records += len(records_chunk)
            total_sentences += len(chunk_items)

            elapsed = max(0.001, time.time() - stream_start)
            rate_rec = total_records / elapsed
            rate_sent = total_sentences / elapsed
            if estimated_total and estimated_total > 0:
                pct = (total_records * 100.0) / estimated_total
                rem_sec = max(
                    0.0, (estimated_total - total_records) / max(1.0, rate_rec)
                )
                eta_text = f"ETA: {format_eta(rem_sec)}"
                pct_text = f"{pct:.1f}% ({total_records:,}/{estimated_total:,})"
            else:
                eta_text = ""
                pct_text = f"{total_records:,} records"

            print(
                f"  Progress: {pct_text} | {rate_rec:.0f} recs/sec ({rate_sent:.0f} sents/sec) | "
                f"Cache: {len(cache)} choices (+{new_choices_added}) | {eta_text}".rstrip(
                    " |"
                ),
                flush=True,
            )

    finally:
        write_queue.put(write_sentinel)
        writer_thread.join()
        if write_err:
            raise write_err[0]
        if out_f is not None:
            out_f.close()
        if parquet_writer is not None:
            parquet_writer.close()

    total_elapsed = max(0.001, time.time() - start_time)

    export_metadata = {
        "language": lang,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "corpus_input": str(input_source),
        "corpus_revision": dataset_revision,
        "corpus_split": dataset_split,
        "model": str(effective_model_id),
        "model_revision": rev,
        "seed_cache_sha256": (
            hashlib.sha256(resolved_cache_path.read_bytes()).hexdigest()
            if resolved_cache_path
            else None
        ),
        "preserve_unreviewed": preserve_unreviewed,
        "prompt_words": prompt_words,
        "total_choices": len(cache),
        "total_records_processed": total_records,
        "total_sentences_processed": total_sentences,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_elapsed_hours": round(total_elapsed / 3600.0, 2),
        "total_elapsed_human": format_eta(total_elapsed),
        "overlong_sentences_preserved_unchanged": overlong_sentences_skipped,
    }

    print("\n[Phase 2/2] 💾 Exporting Unified Sealed Cache...", flush=True)
    export_sealed_cache(
        choices=cache,
        output_path=sealed_path,
        metadata=export_metadata,
        corpus_path=output_path,
        text_field=text_field,
        normalized_field=output_field,
        lang=lang,
        decision_stats=decision_stats,
        invalid_pairs=invalid_pairs,
    )
    print(f"  ✓ Saved unified sealed cache to: {sealed_path}", flush=True)
    return 0

"""Unified command-line interface for NormForm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

from .audit import extract_changed_alternatives, find_inconsistencies
from .batch import is_huggingface_dataset, run_batch_standardization
from .evidence import find_representative_evidence
from .harness import (
    DEFAULT_MODEL_BY_LANG,
    ChoiceCache,
    DocumentHarness,
    Standardizer,
    parse_prompt_words,
    pinned_model_identifier,
)
from .model import load_adapter
from .styles import (
    DEFAULT_STYLE_REPOS,
    DEFAULT_STYLE_REVISIONS,
    STYLE_ALIASES,
    load_style_profile,
    pull_style,
    push_style,
)


def _cmd_standardize(args: argparse.Namespace) -> int:
    text_in = args.text or (
        args.target if args.target and not Path(args.target).exists() and not is_huggingface_dataset(args.target) else None
    )
    input_source = args.input or (args.target if text_in is None else None)

    if args.batch or (
        input_source
        and (
            is_huggingface_dataset(str(input_source))
            or Path(str(input_source)).is_dir()
            or Path(str(input_source)).suffix.lower() == ".parquet"
        )
    ):
        if not input_source:
            print("Error: --batch requires an input file, directory, or Hugging Face dataset.", file=sys.stderr)
            return 2
        return _run_batch(args, input_source)

    model_id = args.model or DEFAULT_MODEL_BY_LANG[args.lang]
    style_profile = load_style_profile(args.style, lang=args.lang) if args.style else None
    adapter, config = load_adapter(
        args.config, model_id, device_name=args.device, require_multitask=True
    )
    identifier = (
        f"{config['model']['id']}@{config['model']['revision']}:"
        f"{pinned_model_identifier(model_id)}"
    )

    temp_dir = None
    if args.cache:
        cache = ChoiceCache(
            args.cache,
            model_identifier=identifier,
            allow_model_mismatch=args.allow_model_mismatch,
        )
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="normform_cache_")
        cache = ChoiceCache(
            Path(temp_dir.name) / "style-cache-ephemeral.json",
            model_identifier=identifier,
        )

    words = parse_prompt_words(args.prompt_words)
    harness = DocumentHarness(
        adapter,
        cache,
        style_profile=style_profile,
        preserve_unreviewed=args.preserve_unreviewed,
        max_generations=args.max_generations,
        max_temperature=args.max_temperature,
        debug=args.debug,
    )

    try:
        if text_in and not input_source:
            result = harness.process_record(
                {"text": text_in},
                record_index=0,
                default_prompt_words=words,
                include_trace=args.include_trace,
            )
            std_text = result["standardised_text"]
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if args.output.suffix.lower() == ".jsonl":
                    args.output.write_text(
                        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                else:
                    args.output.write_text(std_text + "\n", encoding="utf-8")
            else:
                print(std_text)
        elif input_source:
            input_path = Path(str(input_source))
            if not input_path.is_file():
                print(f"Error: Input file not found: {input_path}", file=sys.stderr)
                return 1
            in_ext = input_path.suffix.lower()
            is_txt = in_ext == ".txt"
            output_path = args.output or input_path.with_name(
                f"{input_path.stem}.standardised{in_ext}"
            )
            output_is_txt = output_path.suffix.lower() == ".txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            input_field = args.input_field or "text"
            std_out_field = f"standardised_{input_field}"

            with (
                input_path.open(encoding="utf-8") as source,
                output_path.open("w", encoding="utf-8") as target,
            ):
                for index, line in enumerate(source):
                    if not line.strip():
                        continue
                    record = (
                        {"text": line.rstrip("\n")} if is_txt else json.loads(line)
                    )
                    result = harness.process_record(
                        record,
                        record_index=index,
                        default_prompt_words=words,
                        include_trace=args.include_trace,
                        text_field=input_field,
                    )
                    if output_is_txt:
                        target.write(result[std_out_field] + "\n")
                    else:
                        target.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[normform] Saved output to '{output_path}'.", file=sys.stderr)
        else:
            if not sys.stdin.isatty():
                stdin_text = sys.stdin.read().strip()
                result = harness.process_record(
                    {"text": stdin_text},
                    record_index=0,
                    default_prompt_words=words,
                    include_trace=args.include_trace,
                )
                print(result["standardised_text"])
            else:
                print(
                    "Error: Provide input text (-t / positional) or an input file (-i).",
                    file=sys.stderr,
                )
                return 2

        if args.summary or args.summary_output:
            summary = harness.summary.markdown()
            if args.summary:
                print(summary, end="")
            if args.summary_output:
                args.summary_output.parent.mkdir(parents=True, exist_ok=True)
                args.summary_output.write_text(summary, encoding="utf-8")
    finally:
        if temp_dir:
            temp_dir.cleanup()
    return 0


def _run_batch(args: argparse.Namespace, input_source: str | Path) -> int:
    src_str = str(input_source)
    if not is_huggingface_dataset(src_str):
        in_path = Path(src_str)
        if not in_path.exists():
            print(f"Error: Input not found: {src_str}", file=sys.stderr)
            return 1
        output_path = args.output or (
            in_path.parent / f"{in_path.stem}_normalized{in_path.suffix or '.jsonl'}"
        )
    else:
        clean_id = src_str.removeprefix("hf://").replace("/", "_")
        output_path = args.output or Path(f"./{clean_id}_normalized.jsonl")

    sealed_path = getattr(args, "sealed_cache_output", None) or (
        output_path.parent / f"{output_path.stem}_sealed_cache.yaml"
    )
    return run_batch_standardization(
        input_source=input_source,
        output_path=Path(output_path),
        sealed_path=Path(sealed_path),
        lang=args.lang,
        style=getattr(args, "style", None),
        seed_cache_path=getattr(args, "seed_cache", None),
        prompt_words_arg=getattr(args, "prompt_words", None),
        preserve_unreviewed=getattr(args, "preserve_unreviewed", False),
        text_field=getattr(args, "input_field", None) or getattr(args, "text_field", "text"),
        output_field=getattr(args, "output_field", "normalized"),
        dataset_split=getattr(args, "dataset_split", "train"),
        dataset_revision=getattr(args, "dataset_revision", None),
        batch_size=getattr(args, "batch_size", 512),
        chunk_size=getattr(args, "chunk_size", 4096),
        limit=getattr(args, "limit", None),
        model_id=args.model,
        config_path=args.config,
        device=args.device,
    )


def _cmd_batch(args: argparse.Namespace) -> int:
    input_source = args.input or args.target
    if not input_source:
        print("Error: Provide an input file, directory, or Hugging Face dataset.", file=sys.stderr)
        return 2
    return _run_batch(args, input_source)


def _cmd_alternatives(args: argparse.Namespace) -> int:
    text_in = args.text or (
        args.target if args.target and not Path(args.target).exists() else None
    )
    input_path = Path(args.input) if args.input else (
        Path(args.target) if args.target and text_in is None else None
    )
    std = Standardizer(
        lang=args.lang,
        config_path=args.config,
        model_identifier=args.model,
        device=args.device,
    )
    if text_in and not input_path:
        out = std.alternatives(text_in)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(out + "\n", encoding="utf-8")
        else:
            print(out)
        return 0
    if input_path:
        if not input_path.is_file():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            return 1
        lines = input_path.read_text(encoding="utf-8").splitlines()
        out_lines = [std.alternatives(line) if line.strip() else line for line in lines]
        rendered = "\n".join(out_lines) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    if not sys.stdin.isatty():
        print(std.alternatives(sys.stdin.read().strip()))
        return 0
    print("Error: Provide text (-t / positional) or an input file (-i).", file=sys.stderr)
    return 2


def _cmd_audit(args: argparse.Namespace) -> int:
    input_arg = args.input or args.target
    if args.stats:
        if not input_arg or not Path(input_arg).is_file():
            print("Error: --stats requires an existing standardized corpus file (.jsonl or .parquet).", file=sys.stderr)
            return 2
        ranked, total_scanned, total_changed = extract_changed_alternatives(
            corpus_path=Path(input_arg),
            text_field=args.text_field,
            normalized_field=args.normalized_field,
            lang=args.lang,
            limit=args.limit,
        )
        pct = (total_changed / max(1, total_scanned)) * 100.0 if total_scanned else 0.0
        print(f"# Scanned: {total_scanned:,} | Changed: {total_changed:,} ({pct:.2f}%)")
        print("choices:")
        top_items = ranked[: args.top] if args.top else ranked
        for grp, chosen, tot, details in top_items:
            detail_str = ", ".join(f"{w}: {c}" for w, c in details)
            print(f"  {' | '.join(grp)}: {chosen}  # total: {tot} ({detail_str})")
        return 0

    records: list[dict[str, Any]] = []
    if input_arg:
        in_path = Path(input_arg)
        if in_path.is_file():
            if in_path.suffix.lower() == ".jsonl":
                with in_path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            else:
                with in_path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append({"text": line.rstrip("\n")})
        else:
            records.append({"text": str(input_arg)})
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            if line.strip():
                records.append({"text": line.rstrip("\n")})
    else:
        print("Error: Provide input text or file (-i) to audit.", file=sys.stderr)
        return 2

    model_name = args.model or DEFAULT_MODEL_BY_LANG[args.lang]
    style_profile = load_style_profile(args.style, lang=args.lang) if args.style else None
    adapter, config = load_adapter(args.config, model_name, device_name=args.device)
    identifier = f"{config['model']['id']}@{config['model']['revision']}:{model_name}"

    with tempfile.TemporaryDirectory(prefix="normform_audit_") as tmp_dir:
        cache = ChoiceCache(Path(tmp_dir) / "audit-cache.json", model_identifier=identifier)
        harness = DocumentHarness(adapter, cache, style_profile=style_profile)
        inconsistencies = find_inconsistencies(
            records,
            harness,
            parse_prompt_words(args.prompt),
            use_cache=not args.uncached,
            raw_mode=args.raw,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(inconsistencies, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(inconsistencies, ensure_ascii=False, indent=2))
        return 0

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"🔍 NORMFORM INCONSISTENCY REPORT ({len(inconsistencies)} issues found)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    if not inconsistencies:
        print("✅ No morphological choice inconsistencies detected!", file=sys.stderr)
        return 0
    for idx, inc in enumerate(inconsistencies, 1):
        grp = inc["candidate_group"]
        first = inc["first_occurrence"]
        incons = inc["inconsistent_occurrence"]
        print(f"\n⚠️ Issue #{idx}: Inconsistent Choice for Candidate Group {grp}", file=sys.stderr)
        print(f"   Earlier Decision: {first['choice']!r} (Record #{first['record_index']}, Sentence #{first['sentence_index']})", file=sys.stderr)
        print(f"   Later Decision:   {incons['choice']!r} (Record #{incons['record_index']}, Sentence #{incons['sentence_index']})", file=sys.stderr)
    return 0


def _cmd_extract_style(args: argparse.Namespace) -> int:
    input_arg = args.input or args.target
    if input_arg:
        p = Path(input_arg).expanduser()
        doc_text = p.read_text(encoding="utf-8") if p.is_file() else str(input_arg)
    elif not sys.stdin.isatty():
        doc_text = sys.stdin.read()
    else:
        print("Error: Provide an input file or text (-i) to extract style from.", file=sys.stderr)
        return 2

    model_id = args.model or DEFAULT_MODEL_BY_LANG[args.lang]
    adapter, _ = load_adapter(args.config, model_id, device_name=args.device)
    result = find_representative_evidence(
        adapter,
        doc_text,
        n=args.num_words,
        lang=args.lang,
        show_progress=not args.quiet,
    )
    if args.prompt_only:
        print(",".join(result.words))
        return 0
    payload: dict[str, Any] = {
        "prompt_string": ",".join(result.words),
        "explained_variance_pct": round(result.explained_variance, 2),
    }
    if args.verbose:
        payload["words"] = result.words
        payload["total_variance"] = round(result.total_variance, 4)
        payload["step_details"] = result.details
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_styles(args: argparse.Namespace) -> int:
    action = args.styles_action or "list"
    if action == "list":
        print("Official NormForm Style Repositories on Hugging Face:")
        for lang, repo in DEFAULT_STYLE_REPOS.items():
            rev = DEFAULT_STYLE_REVISIONS.get(repo, "main")
            aliases = [k for k, v in STYLE_ALIASES.items() if v == repo]
            print(f"  • [{lang}] {repo}@{rev[:10]}")
            print(f"          Aliases: {', '.join(aliases)}")
        return 0
    if action == "pull":
        if not args.style:
            print("Error: Specify a style alias or Hugging Face repo ID to pull.", file=sys.stderr)
            return 2
        out = args.output or Path("style.yaml")
        saved = pull_style(args.style, out, lang=args.lang, revision=args.revision)
        print(f"Downloaded style '{args.style}' to {saved}")
        return 0
    if action == "push":
        if not args.file or not args.repo:
            print("Error: 'styles push' requires --file <path.yaml> and --repo <org/name>.", file=sys.stderr)
            return 2
        commit = push_style(
            Path(args.file),
            args.repo,
            private=not args.public,
            commit_message=args.message,
        )
        print(f"Published '{args.file}' to Hugging Face dataset '{args.repo}' ({commit})")
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="normform",
        description="NormForm: Context-aware Norwegian morphological standardization, alternative generation, and style auditing.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # 1. standardize
    p_std = subparsers.add_parser(
        "standardize",
        help="Standardize a text string, document (.txt/.jsonl), or dataset.",
    )
    p_std.add_argument("target", nargs="?", default=None, help="Input text string, file path, or HF dataset ID.")
    p_std.add_argument("-i", "--input", type=str, default=None, help="Input file path (.txt, .jsonl, .parquet) or HF dataset.")
    p_std.add_argument("-t", "--text", type=str, default=None, help="Direct text string to standardize.")
    p_std.add_argument("-o", "--output", type=Path, default=None, help="Output file path.")
    p_std.add_argument("-s", "--style", type=str, default=None, help="Style alias ('redaksjonsnorm', 'nasjonalbiblioteknorm'), HF dataset repo ('org/style'), or local .yaml file.")
    p_std.add_argument("-p", "--prompt-words", default="", help="Comma-separated steering prompt words (e.g. 'boka,jenta,kasta').")
    p_std.add_argument("-c", "--cache", type=Path, default=None, help="Persistent JSON document choice cache path.")
    p_std.add_argument("-f", "--input-field", "--text-field", dest="input_field", default=None, help="Input text column/field name (default: 'text').")
    p_std.add_argument("--output-field", default="normalized", help="Output text column name in batch mode (default: 'normalized').")
    p_std.add_argument("-l", "--lang", "--language", dest="lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_std.add_argument("-m", "--model", default=None, help="Override model checkpoint path or HF model ID.")
    p_std.add_argument("-g", "--config", type=Path, default=None, help="Optional runtime model config YAML.")
    p_std.add_argument("-d", "--device", default="auto", help="Execution device ('auto', 'cuda', 'cpu', 'cuda:0').")
    p_std.add_argument("--preserve-unreviewed", action="store_true", help="Preserve source form for candidate groups absent from --style.")
    p_std.add_argument("--batch", action="store_true", help="Use high-throughput streaming GPU batch mode.")
    p_std.add_argument("-b", "--batch-size", type=int, default=512, help="GPU batch size in batch mode (default: 512).")
    p_std.add_argument("--chunk-size", type=int, default=4096, help="Streaming chunk size in batch mode (default: 4096).")
    p_std.add_argument("--limit", type=int, default=None, help="Max records to process in batch mode.")
    p_std.add_argument("--include-trace", action="store_true", help="Include generation trace in JSONL output.")
    p_std.add_argument("--max-generations", type=int, default=5, help="Max sampled retries after greedy pass (1-5).")
    p_std.add_argument("--max-temperature", type=float, default=1.0, help="Final retry temperature.")
    p_std.add_argument("--debug", action="store_true", help="Print detailed generation/cache trace to stderr.")
    p_std.add_argument("--summary", action="store_true", help="Print Markdown summary to stdout.")
    p_std.add_argument("--summary-output", type=Path, default=None, help="Write Markdown summary to file.")
    p_std.add_argument("--allow-model-mismatch", action="store_true", help="Allow reusing a JSON choice cache across models.")

    # 2. batch
    p_batch = subparsers.add_parser(
        "batch",
        help="High-throughput 1-GPU streaming standardization for large corpora (.jsonl, .parquet, HF datasets).",
    )
    p_batch.add_argument("target", nargs="?", default=None, help="Input file, shard directory, or HF dataset ID.")
    p_batch.add_argument("-i", "--input", type=str, default=None, help="Input file, shard directory, or HF dataset ID.")
    p_batch.add_argument("-o", "--output", type=Path, default=None, help="Output file path (.jsonl or .parquet).")
    p_batch.add_argument("-s", "--style", type=str, default=None, help="Style alias, HF dataset repo, or local .yaml file.")
    p_batch.add_argument("--seed-cache", type=Path, default=None, help="Path to local seed cache YAML/JSON.")
    p_batch.add_argument("--sealed-cache-output", type=Path, default=None, help="Output path for completed sealed cache YAML.")
    p_batch.add_argument("--preserve-unreviewed", action="store_true", help="Preserve source form for candidate sets absent from style cache.")
    p_batch.add_argument("-p", "--prompt-words", type=str, default=None, help="Steering prompt words or file path.")
    p_batch.add_argument("-l", "--lang", "--language", dest="lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_batch.add_argument("--dataset-split", default="train", help="HF dataset split (default: train).")
    p_batch.add_argument("--dataset-revision", default=None, help="Pinned HF dataset commit revision.")
    p_batch.add_argument("--text-field", default="text", help="Input text column name (default: text).")
    p_batch.add_argument("--output-field", default="normalized", help="Output normalized column name (default: normalized).")
    p_batch.add_argument("-b", "--batch-size", type=int, default=512, help="GPU batch size (default: 512).")
    p_batch.add_argument("--chunk-size", type=int, default=4096, help="Streaming chunk size (default: 4096).")
    p_batch.add_argument("--limit", type=int, default=None, help="Process only first N records.")
    p_batch.add_argument("-m", "--model", default=None, help="Override model checkpoint or HF repo ID.")
    p_batch.add_argument("-c", "--config", type=Path, default=None, help="Optional model config YAML.")
    p_batch.add_argument("-d", "--device", default="auto", help="Execution device ('auto', 'cuda:0', 'cpu').")

    # 3. alternatives
    p_alt = subparsers.add_parser(
        "alternatives",
        help="Generate contextually valid [alt1|alt2] morphological brackets for text.",
    )
    p_alt.add_argument("target", nargs="?", default=None, help="Input text string or file path.")
    p_alt.add_argument("-i", "--input", type=str, default=None, help="Input file path.")
    p_alt.add_argument("-t", "--text", type=str, default=None, help="Direct text string.")
    p_alt.add_argument("-o", "--output", type=Path, default=None, help="Output file path.")
    p_alt.add_argument("-l", "--lang", "--language", dest="lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_alt.add_argument("-m", "--model", default=None, help="Override model checkpoint or HF repo ID.")
    p_alt.add_argument("-c", "--config", type=Path, default=None, help="Optional model config YAML.")
    p_alt.add_argument("-d", "--device", default="auto", help="Execution device.")

    # 4. audit
    p_audit = subparsers.add_parser(
        "audit",
        help="Audit documents for morphological inconsistencies or report corpus change statistics.",
    )
    p_audit.add_argument("target", nargs="?", default=None, help="Input file or text string.")
    p_audit.add_argument("-i", "--input", type=str, default=None, help="Input file or text string.")
    p_audit.add_argument("-s", "--style", type=str, default=None, help="Style alias, HF repo ID, or local .yaml file.")
    p_audit.add_argument("-p", "--prompt", default="", help="Comma-separated prompt words.")
    p_audit.add_argument("-l", "--lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_audit.add_argument("-m", "--model", default=None, help="Override model checkpoint or HF repo ID.")
    p_audit.add_argument("-c", "--config", type=Path, default=None, help="Optional model config YAML.")
    p_audit.add_argument("-d", "--device", default="auto", help="Execution device.")
    p_audit.add_argument("-o", "--output", type=Path, default=None, help="Write JSON report to path.")
    p_audit.add_argument("--raw", "--source-only", action="store_true", help="Audit raw source text for internal inconsistencies.")
    p_audit.add_argument("--uncached", action="store_true", help="Disable cross-sentence choice cache to test raw model stability.")
    p_audit.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    p_audit.add_argument("--stats", action="store_true", help="Extract ranked changed-alternative statistics from a standardized corpus.")
    p_audit.add_argument("--text-field", default="text", help="Source text field for --stats (default: text).")
    p_audit.add_argument("--normalized-field", default="normalized", help="Normalized text field for --stats (default: normalized).")
    p_audit.add_argument("--top", type=int, default=50, help="Top N changed groups to display for --stats (0 = all).")
    p_audit.add_argument("--limit", type=int, default=None, help="Max records to scan for --stats.")

    # 5. extract-style
    p_ext = subparsers.add_parser(
        "extract-style",
        help="Extract representative steering prompt words from a reference document.",
    )
    p_ext.add_argument("target", nargs="?", default=None, help="Input file or text string.")
    p_ext.add_argument("-i", "--input", type=str, default=None, help="Input file or text string.")
    p_ext.add_argument("-l", "--lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_ext.add_argument("-n", "--num-words", type=int, default=10, help="Number of prompt words to extract (default: 10).")
    p_ext.add_argument("-m", "--model", default=None, help="Override model checkpoint or HF repo ID.")
    p_ext.add_argument("-c", "--config", type=Path, default=None, help="Optional model config YAML.")
    p_ext.add_argument("-d", "--device", default="auto", help="Execution device.")
    p_ext.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    p_ext.add_argument("-v", "--verbose", action="store_true", help="Include step-by-step likelihood gains in JSON output.")
    p_ext.add_argument("-p", "--prompt-only", action="store_true", help="Print only comma-separated prompt words.")

    # 6. styles
    p_styles = subparsers.add_parser(
        "styles",
        help="List, pull, or push NormForm YAML style caches on Hugging Face.",
    )
    p_styles.add_argument("styles_action", nargs="?", choices=["list", "pull", "push"], default="list", help="Action: list, pull, or push.")
    p_styles.add_argument("style", nargs="?", default=None, help="Style alias or HF repo ID for 'pull'.")
    p_styles.add_argument("-f", "--file", type=Path, default=None, help="Local YAML style file for 'push'.")
    p_styles.add_argument("-r", "--repo", type=str, default=None, help="Target HF dataset repo ID ('org/name') for 'push'.")
    p_styles.add_argument("-o", "--output", type=Path, default=None, help="Destination file path for 'pull' (default: ./style.yaml).")
    p_styles.add_argument("-l", "--lang", default="nob", choices=["nob", "nno"], help="Language ('nob' or 'nno').")
    p_styles.add_argument("--revision", default=None, help="Pinned HF commit revision for 'pull'.")
    p_styles.add_argument("--public", action="store_true", help="Create public HF dataset repo on 'push' (default: private).")
    p_styles.add_argument("-m", "--message", default="Publish NormForm style cache", help="Commit message for 'push'.")

    return parser


KNOWN_COMMANDS = {"standardize", "batch", "alternatives", "audit", "extract-style", "styles"}


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    # Allow shorthand `normform -t "..."` or `normform -i file.txt` without typing `standardize`
    if raw_args and raw_args[0] not in KNOWN_COMMANDS and raw_args[0] not in ("-h", "--help"):
        raw_args = ["standardize"] + raw_args

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.command == "standardize":
        return _cmd_standardize(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "alternatives":
        return _cmd_alternatives(args)
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "extract-style":
        return _cmd_extract_style(args)
    if args.command == "styles":
        return _cmd_styles(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run a prompt-conditioned NormForm model over documents with a shared cache or style profile."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import difflib
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata

from .contract import Template, clean_direct, parse
from .model import (
    DEFAULT_MODEL_BY_LANG,
    DEFAULT_MODEL_ID,
    DEFAULT_NNO_MODEL_ID,
    DEFAULT_NNO_MODEL_REVISION,
    DEFAULT_NOB_MODEL_ID,
    DEFAULT_NOB_MODEL_REVISION,
    load_adapter,
    model_revision,
    pinned_model_identifier,
)
from .styles import (
    CACHE_SCHEMA,
    ChoiceCache,
    StyleProfile,
    canonical_choice_key,
    choice_key,
    is_invalid_pair,
    load_style_profile,
    resolve_candidate_choice,
)

_BOUNDARY = re.compile(r"(?<=[.!?…])(?=\s|$)")


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split on terminal punctuation while preserving every character and offset."""
    if not text:
        return []
    spans: list[SentenceSpan] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if text[start:end]:
            spans.append(SentenceSpan(start, end, text[start:end]))
        start = end
    if start < len(text):
        spans.append(SentenceSpan(start, len(text), text[start:]))
    if len(spans) > 1 and spans[-1].text.isspace():
        trailing = spans.pop()
        previous = spans[-1]
        spans[-1] = SentenceSpan(
            previous.start, trailing.end, previous.text + trailing.text
        )
    return spans


def parse_prompt_words(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _word_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def forced_choice(
    candidates: Sequence[str],
    prompt_words: Sequence[str],
    errors: list[str] | None = None,
) -> str | None:
    """Return the candidate named by a prompt word, reporting ambiguity."""
    normalized_words = [unicodedata.normalize("NFC", word) for word in prompt_words]
    exact_matches = [
        candidate
        for candidate in candidates
        if unicodedata.normalize("NFC", candidate) in normalized_words
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    matches = [
        candidate
        for candidate in candidates
        if _word_key(candidate) in {_word_key(word) for word in prompt_words}
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    selected = max(
        enumerate(matches),
        key=lambda item: max(
            difflib.SequenceMatcher(None, item[1], word).ratio()
            for word in normalized_words
        ),
    )[1]
    message = (
        "Prompt word is ambiguous within one alternative group: "
        f"{list(candidates)!r} matches {matches!r}; selected closest {selected!r}"
    )
    if errors is not None:
        errors.append(message)
    return selected


def _render_with_locks(template: Template, locked: Mapping[int, str]) -> str:
    choices = []
    for index, candidates in enumerate(template.groups):
        choices.append(
            locked[index] if index in locked else "[" + "|".join(candidates) + "]"
        )
    parts = [template.literals[0]]
    for index, value in enumerate(choices):
        parts.extend((value, template.literals[index + 1]))
    return "".join(parts)


class HarnessSummary:
    """Counts collected across one harness run, rendered as Markdown."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount

    def markdown(self) -> str:
        c = self.counts
        lines = [
            "# NormForm harness summary",
            "",
            "| Measure | Count |",
            "|---|---:|",
            f"| Input records | {c['records']} |",
            f"| Documents changed | {c['changed_records']} |",
            f"| Documents unchanged | {c['unchanged_records']} |",
            f"| Sentence spans | {c['sentences']} |",
            f"| Non-blank sentences | {c['nonblank_sentences']} |",
            f"| Sentences changed | {c['changed_sentences']} |",
            f"| Sentences unchanged | {c['unchanged_sentences']} |",
            f"| Blank sentence spans | {c['blank_sentences']} |",
            f"| Alternative lattices accepted | {c['alternatives_accepted']} |",
            f"| Alternative-mode original fallbacks | {c['alternatives_fallback']} |",
            f"| Alternative generations tried | {c['alternative_attempts']} |",
            f"| Invalid alternative generations | {c['invalid_alternative_attempts']} |",
            f"| Style generations tried | {c['style_attempts']} |",
            f"| Valid style generations accepted | {c['style_accepted']} |",
            f"| Style greedy fallbacks | {c['style_fallback']} |",
            f"| Style generations skipped (already resolved) | {c['style_skipped']} |",
            f"| Invalid style generations | {c['invalid_style_attempts']} |",
            f"| Prompt-locked groups | {c['prompt_locked_groups']} |",
            f"| Cache-locked groups | {c['cache_locked_groups']} |",
            f"| Choices written to cache | {c['choices_written']} |",
            f"| Harness errors | {c['errors']} |",
            "",
        ]
        return "\n".join(lines)


def _strip_ws(text: str) -> tuple[str, str, str]:
    leading_len = len(text) - len(text.lstrip())
    leading = text[:leading_len]
    rest = text[leading_len:]
    trailing_len = len(rest) - len(rest.rstrip())
    if trailing_len > 0:
        core = rest[:-trailing_len]
        trailing = rest[-trailing_len:]
    else:
        core = rest
        trailing = ""
    return leading, core, trailing


class DocumentHarness:
    def __init__(
        self,
        adapter: Any,
        cache: ChoiceCache,
        *,
        style_profile: StyleProfile | None = None,
        preserve_unreviewed: bool = False,
        max_generations: int = 5,
        max_temperature: float = 1.0,
        debug: bool = False,
    ):
        if not 1 <= max_generations <= 5:
            raise ValueError("max_generations must be between 1 and 5")
        if not 0.0 <= max_temperature <= 1.0:
            raise ValueError("max_temperature must be between 0 and 1")
        self.adapter = adapter
        self.cache = cache
        self.style_profile = style_profile
        self.preserve_unreviewed = preserve_unreviewed
        self.max_generations = max_generations
        self.max_temperature = max_temperature
        self.debug = debug
        self.summary = HarnessSummary()

    def _temperatures(self) -> list[float]:
        """Return sampled retry temperatures; greedy is handled separately."""
        step = self.max_temperature / self.max_generations
        return [
            round((index + 1) * step, 6) for index in range(self.max_generations)
        ]

    def _call(
        self,
        method: Any,
        *args: Any,
        temperature: float,
        strict: bool = False,
    ) -> Any:
        """Call old test adapters as well as temperature-aware model adapters."""
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        names = {parameter.name for parameter in parameters}
        kwargs: dict[str, Any] = {}
        if accepts_kwargs or "temperature" in names:
            kwargs["temperature"] = temperature
        if strict and (accepts_kwargs or "strict" in names):
            kwargs["strict"] = True
        return method(*args, **kwargs)

    def _debug_event(self, event: Mapping[str, Any]) -> None:
        if not self.debug:
            return

        CYAN = "\033[36m\033[1m"
        GREEN = "\033[32m\033[1m"
        RED = "\033[31m\033[1m"
        YELLOW = "\033[33m\033[1m"
        MAGENTA = "\033[35m"
        RESET = "\033[0m"

        phase = event.get("phase")
        decision = event.get("decision")
        attempt = event.get("attempt")
        mode = event.get("mode")
        temp = event.get("temperature")
        valid = event.get("valid")
        reason = event.get("reason")
        raw = event.get("raw")

        lines: list[str] = []

        if phase == "sentence_start":
            rec_idx = event.get("record")
            sent_idx = event.get("sentence")
            text = event.get("text", "")
            lines.append(
                f"\n📄 Record #{rec_idx} | Sentence #{sent_idx}: {CYAN}{text!r}{RESET}"
            )

        elif phase == "alternatives":
            if mode is not None:
                mode_str = f"{mode}, temp={temp}"
                if valid:
                    status_str = f"{GREEN}✅ VALID{RESET} (Lattice aligned)"
                else:
                    status_str = f"{RED}❌ INVALID{RESET} ({YELLOW}{reason}{RESET})"
                lines.append(
                    f"  │  ├─ 🔄 [Alternatives] Attempt {attempt} ({mode_str}): Raw -> {MAGENTA}{raw!r}{RESET}"
                )
                lines.append(f"  │  │  └─ Status: {status_str}")
            elif decision:
                if decision == "accepted":
                    lines.append(f"  │  └─ Decision: {GREEN}✅ ACCEPTED lattice{RESET}")
                else:
                    lines.append(
                        f"  │  └─ Decision: {YELLOW}⚠️ FALLBACK to original text{RESET} after {event.get('attempts')} attempts"
                    )

        elif phase == "groups":
            group_idx = event.get("group")
            candidates = event.get("candidates")
            status = event.get("status")
            choice = event.get("choice")
            cand_str = "[" + "|".join(candidates) + "]"
            if status == "prompt":
                lines.append(
                    f"  │  ├─ 📌 [PROMPT LOCK] Group #{group_idx} {cand_str} -> Locked to {GREEN}{choice!r}{RESET}"
                )
            elif status == "cache":
                lines.append(
                    f"  │  ├─ 💾 [DB READ] Group #{group_idx} {cand_str} -> Retrieved {GREEN}{choice!r}{RESET} from cache"
                )
            else:
                lines.append(
                    f"  │  ├─ 🔓 [OPEN] Group #{group_idx} {cand_str} -> Awaiting model decision"
                )

        elif phase == "cache_write":
            writes = event.get("writes", [])
            cache_name = event.get("cache_name", "cache.json")
            lines.append(
                f"  │  ├─ 💾 [DB WRITE] Saved {len(writes)} choices to {CYAN}{cache_name}{RESET}:"
            )
            for i, item in enumerate(writes):
                cand_str = "[" + "|".join(item["candidates"]) + "]"
                prefix = "  │  │  └─" if i == len(writes) - 1 else "  │  │  ├─"
                lines.append(
                    f"{prefix} Group #{item['group']} {cand_str} -> {GREEN}{item['choice']!r}{RESET}"
                )

        elif phase == "style":
            if decision == "already_resolved":
                cache_locks = event.get("cache_locked_groups", 0)
                prompt_locks = event.get("prompt_locked_groups", 0)
                choices = event.get("choices", [])
                lines.append(
                    f"  │  └─ ⚡ [Style] SKIPPED: All groups resolved ({cache_locks} from cache, {prompt_locks} from prompts)"
                )
                lines.append(f"  │     └─ Choices: {GREEN}{choices}{RESET}")
            elif mode is not None:
                mode_str = f"{mode}, temp={temp}"
                if valid:
                    choices = event.get("choices", [])
                    status_str = (
                        f"{GREEN}✅ VALID{RESET} (choices: {GREEN}{choices}{RESET})"
                    )
                else:
                    status_str = f"{RED}❌ INVALID{RESET} ({YELLOW}{reason}{RESET})"
                lines.append(
                    f"  │  ├─ 🤖 [Style] Attempt {attempt} ({mode_str}): Raw -> {MAGENTA}{raw!r}{RESET}"
                )
                lines.append(f"  │  │  └─ Status: {status_str}")
            elif decision:
                text = event.get("text", "")
                if decision == "accepted":
                    lines.append(
                        f"  │  └─ Decision: {GREEN}✅ ACCEPTED{RESET} -> {CYAN}{text!r}{RESET}"
                    )
                else:
                    lines.append(
                        f"  │  └─ Decision: {YELLOW}⚠️ GREEDY FALLBACK{RESET} -> {CYAN}{text!r}{RESET}"
                    )

        elif phase == "sentence_end":
            text = event.get("text", "")
            changed = event.get("changed", False)
            color = GREEN if changed else CYAN
            lines.append(f"  └─ ✨ Final Output: {color}{text!r}{RESET}")

        if lines:
            for line in lines:
                print(line, file=sys.stderr, flush=True)

        if phase not in ("sentence_start", "sentence_end", "groups"):
            print(
                "[harness] "
                + json.dumps(dict(event), ensure_ascii=False, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    def _alternative_attempts(
        self, source: str, *, record_index: int, sentence_index: int
    ) -> tuple[str, list[dict[str, Any]]]:
        leading, core_source, trailing = _strip_ws(source)
        if not core_source:
            return source, []
        attempts: list[dict[str, Any]] = []
        temperatures = [0.0] + self._temperatures()
        for attempt, temperature in enumerate(temperatures):
            result = self._call(
                self.adapter.alternatives,
                core_source,
                temperature=temperature,
                strict=True,
            )
            raw = result.get("raw_proposal", result.get("text"))
            valid = True
            reason = "valid_lattice"
            if result.get("source_copy_fallback"):
                valid = False
                reason = "outside_lattice_or_invalid_alternative_output"
            elif "raw_proposal" in result:
                try:
                    if not isinstance(raw, str):
                        raise ValueError("missing raw proposal")
                    canonical = clean_direct(core_source, raw)
                    parse(canonical).align(core_source)
                except (TypeError, ValueError) as exc:
                    valid = False
                    reason = str(exc)
            event = {
                "record": record_index,
                "sentence": sentence_index,
                "phase": "alternatives",
                "attempt": attempt,
                "mode": "greedy" if attempt == 0 else "sampled",
                "temperature": temperature,
                "raw": raw,
                "result": result.get("text"),
                "valid": valid,
                "reason": reason,
            }
            attempts.append(event)
            self._debug_event(event)
            if valid:
                conclusion = {
                    "record": record_index,
                    "sentence": sentence_index,
                    "phase": "alternatives",
                    "decision": "accepted",
                    "attempt": attempt,
                    "temperature": temperature,
                }
                self._debug_event(conclusion)
                res_text = str(result["text"])
                if not res_text.startswith(leading):
                    res_text = leading + res_text
                if not res_text.endswith(trailing):
                    res_text = res_text + trailing
                return res_text, attempts
        conclusion = {
            "record": record_index,
            "sentence": sentence_index,
            "phase": "alternatives",
            "decision": "original_fallback",
            "attempts": len(attempts),
        }
        self._debug_event(conclusion)
        return source, attempts

    def _style_attempts(
        self,
        words: Sequence[str],
        conditioned: str,
        original: str,
        template: Template,
        forced: Mapping[int, str],
        *,
        record_index: int,
        sentence_index: int,
    ) -> tuple[str, tuple[str, ...] | None, list[dict[str, Any]]]:
        leading, core_conditioned, trailing = _strip_ws(conditioned)
        core_template = parse(core_conditioned)

        attempts: list[dict[str, Any]] = []
        greedy_fallback = original
        temperatures = [0.0] + self._temperatures()
        for attempt, temperature in enumerate(temperatures):
            result = self._call(
                self.adapter.style,
                words,
                core_conditioned,
                temperature=temperature,
                strict=attempt > 0,
            )
            if "raw_proposal" in result and result.get("raw_proposal") is not None:
                raw = result["raw_proposal"]
            else:
                raw = result.get("text")
            valid = True
            reason = "valid_lattice"
            choices: tuple[str, ...] | None = None
            resolved_text = result.get("text")
            try:
                if isinstance(resolved_text, str):
                    parse(core_conditioned).align(resolved_text)
                    resolved_choices = core_template.align(resolved_text)
                    for index, choice in forced.items():
                        if resolved_choices[index] != choice:
                            raise ValueError("greedy fallback changed a forced choice")
                    greedy_fallback = (
                        leading + core_template.render(resolved_choices) + trailing
                    )
            except (TypeError, ValueError):
                pass
            try:
                if not isinstance(raw, str):
                    raise ValueError("missing raw proposal")
                parse(core_conditioned).align(raw)
                choices = core_template.align(raw)
                for index, choice in forced.items():
                    if choices[index] != choice:
                        raise ValueError(
                            f"forced choice changed at group {index}: "
                            f"expected {choice!r}, got {choices[index]!r}"
                        )
            except (TypeError, ValueError) as exc:
                valid = False
                reason = str(exc)
            event = {
                "record": record_index,
                "sentence": sentence_index,
                "phase": "style",
                "attempt": attempt,
                "mode": "greedy" if attempt == 0 else "sampled",
                "temperature": temperature,
                "raw": raw,
                "result": result.get("text"),
                "valid": valid,
                "reason": reason,
            }
            if choices is not None:
                event["choices"] = list(choices)
            attempts.append(event)
            self._debug_event(event)
            if not valid or choices is None:
                continue

            by_key: dict[str, str] = {}
            for index, candidates in enumerate(core_template.groups):
                key = choice_key(candidates)
                by_key.setdefault(key, choices[index])
            normalized = tuple(
                by_key[choice_key(candidates)] for candidates in core_template.groups
            )
            final_text = leading + core_template.render(normalized) + trailing
            conclusion = {
                "record": record_index,
                "sentence": sentence_index,
                "phase": "style",
                "decision": "accepted",
                "attempt": attempt,
                "temperature": temperature,
                "text": final_text,
                "choices": list(normalized),
            }
            self._debug_event(conclusion)
            return final_text, normalized, attempts

        conclusion = {
            "record": record_index,
            "sentence": sentence_index,
            "phase": "style",
            "decision": "greedy_fallback",
            "attempts": len(attempts),
            "text": greedy_fallback,
        }
        self._debug_event(conclusion)
        return greedy_fallback, None, attempts

    def process_record(
        self,
        record: Mapping[str, Any],
        *,
        record_index: int,
        default_prompt_words: Sequence[str],
        include_trace: bool = False,
        text_field: str = "text",
    ) -> dict[str, Any]:
        if not isinstance(record.get(text_field), str):
            raise ValueError(
                f"Each JSONL record must contain a '{text_field}' string field"
            )
        base_words = (
            default_prompt_words
            if default_prompt_words
            else (self.style_profile.prompt_words if self.style_profile else [])
        )
        words = parse_prompt_words(record.get("prompt_words", base_words))
        source = record[text_field]
        spans = split_sentences(source)
        self.summary.add("records")
        self.summary.add("sentences", len(spans))
        outputs: list[str] = []
        trace: list[dict[str, Any]] = []
        errors: list[str] = []
        with self.cache.transaction():
            for sentence_index, span in enumerate(spans):
                if not span.text.strip():
                    self.summary.add("blank_sentences")
                    outputs.append(span.text)
                    continue
                self.summary.add("nonblank_sentences")
                self._debug_event(
                    {
                        "phase": "sentence_start",
                        "record": record_index,
                        "sentence": sentence_index,
                        "text": span.text.strip(),
                    }
                )
                alt_text, alternative_attempts = self._alternative_attempts(
                    span.text,
                    record_index=record_index,
                    sentence_index=sentence_index,
                )
                alternatives_valid = bool(
                    alternative_attempts and alternative_attempts[-1]["valid"]
                )
                self.summary.add("alternative_attempts", len(alternative_attempts))
                self.summary.add(
                    "invalid_alternative_attempts",
                    sum(not bool(item["valid"]) for item in alternative_attempts),
                )
                self.summary.add(
                    "alternatives_accepted"
                    if alternatives_valid
                    else "alternatives_fallback"
                )
                if not alternatives_valid:
                    errors.append(
                        f"Sentence {sentence_index}: no valid alternative lattice "
                        f"after {len(alternative_attempts)} attempts; used original"
                    )
                try:
                    template = parse(alt_text)
                    original_choices = template.align(span.text)
                except ValueError as exc:
                    self.summary.add("unchanged_sentences")
                    outputs.append(span.text)
                    errors.append(
                        f"Sentence {sentence_index}: invalid alternative lattice: {exc}"
                    )
                    if include_trace:
                        trace.append(
                            {
                                "source": span.text,
                                "altmorph": alt_text,
                                "standardised": span.text,
                                "alternative_attempts": alternative_attempts,
                                "alternative_decision": "original_fallback",
                                "decision": "original_fallback",
                            }
                        )
                    continue
                locked: dict[int, str] = {}
                forced: dict[int, str] = {}
                for index, candidates in enumerate(template.groups):
                    orig_choice = original_choices[index]
                    if self.style_profile is not None and is_invalid_pair(
                        candidates, self.style_profile.invalid_pairs
                    ):
                        locked[index] = orig_choice
                        forced[index] = orig_choice
                    else:
                        absolute = forced_choice(candidates, words, errors)
                        if absolute is not None:
                            forced[index] = absolute
                            locked[index] = absolute
                        elif self.style_profile is not None:
                            ckey = canonical_choice_key(candidates)
                            profile_choice = self.style_profile.choices.get(ckey)
                            if profile_choice is not None:
                                resolved, hit = resolve_candidate_choice(
                                    candidates,
                                    orig_choice,
                                    profile_choice,
                                    self.style_profile.invalid_pairs,
                                )
                                if hit:
                                    locked[index] = resolved
                        if index not in locked:
                            cached = self.cache.get(candidates)
                            if cached is not None:
                                locked[index] = cached
                            elif self.preserve_unreviewed:
                                locked[index] = orig_choice
                    if self.debug:
                        status = (
                            "prompt"
                            if index in forced
                            else ("cache" if index in locked else "open")
                        )
                        choice = locked.get(index)
                        self._debug_event(
                            {
                                "phase": "groups",
                                "group": index,
                                "candidates": candidates,
                                "status": status,
                                "choice": choice,
                            }
                        )
                self.summary.add("prompt_locked_groups", len(forced))
                self.summary.add("cache_locked_groups", len(locked) - len(forced))
                conditioned = _render_with_locks(template, locked)
                open_groups = [
                    index
                    for index in range(len(template.groups))
                    if index not in locked
                ]
                if not open_groups:
                    final_text = conditioned
                    final_choices = template.align(final_text)
                    style_attempts = []
                    self.summary.add("style_skipped")
                    cache_locks = len(locked) - len(forced)
                    self._debug_event(
                        {
                            "record": record_index,
                            "sentence": sentence_index,
                            "phase": "style",
                            "decision": "already_resolved",
                            "reason": "resolved_by_cache_and_prompts",
                            "cache_locked_groups": cache_locks,
                            "prompt_locked_groups": len(forced),
                            "choices": list(final_choices),
                            "text": final_text,
                        }
                    )
                    decision = (
                        "already_resolved"
                        if alternatives_valid
                        else "original_fallback"
                    )
                else:
                    final_text, final_choices, style_attempts = self._style_attempts(
                        words,
                        conditioned,
                        span.text,
                        template,
                        forced,
                        record_index=record_index,
                        sentence_index=sentence_index,
                    )
                    if final_choices is not None:
                        final_choices = template.align(final_text)
                    self.summary.add("style_attempts", len(style_attempts))
                    self.summary.add(
                        "invalid_style_attempts",
                        sum(not bool(item["valid"]) for item in style_attempts),
                    )
                    self.summary.add(
                        "style_accepted"
                        if final_choices is not None
                        else "style_fallback"
                    )
                    decision = (
                        "accepted"
                        if final_choices is not None
                        else "greedy_fallback"
                    )
                    if final_choices is None:
                        errors.append(
                            f"Sentence {sentence_index}: no valid lattice generation "
                            f"after {len(style_attempts)} attempts; used greedy fallback"
                        )
                if final_choices is not None:
                    new_writes = []
                    for index, candidates in enumerate(template.groups):
                        if index not in forced:
                            new_writes.append(
                                {
                                    "group": index,
                                    "candidates": candidates,
                                    "choice": final_choices[index],
                                }
                            )
                        self.cache.put(
                            candidates,
                            final_choices[index],
                            record_index=record_index,
                            sentence_index=sentence_index,
                            force=index in forced,
                        )
                    self.summary.add("choices_written", len(final_choices))
                    if self.debug and new_writes and decision != "already_resolved":
                        self._debug_event(
                            {
                                "phase": "cache_write",
                                "record": record_index,
                                "sentence": sentence_index,
                                "writes": new_writes,
                                "cache_name": self.cache.path.name,
                            }
                        )
                self.summary.add(
                    "changed_sentences"
                    if final_text != span.text
                    else "unchanged_sentences"
                )
                outputs.append(final_text)
                self._debug_event(
                    {
                        "phase": "sentence_end",
                        "record": record_index,
                        "sentence": sentence_index,
                        "text": final_text.strip(),
                        "changed": final_text != span.text,
                    }
                )
                if include_trace:
                    trace.append(
                        {
                            "source": span.text,
                            "altmorph": alt_text,
                            "conditioned": conditioned,
                            "standardised": final_text,
                            "prompt_words": words,
                            "cached_groups": len(locked),
                            "alternative_attempts": alternative_attempts,
                            "alternative_decision": (
                                "accepted"
                                if alternatives_valid
                                else "original_fallback"
                            ),
                            "style_attempts": style_attempts,
                            "decision": decision,
                        }
                    )
        standardised = "".join(outputs)
        self.summary.add(
            "changed_records" if standardised != source else "unchanged_records"
        )
        self.summary.add("errors", len(errors))
        output = dict(record)
        out_field = f"standardised_{text_field}"
        output[out_field] = standardised
        if include_trace:
            output["harness_trace"] = trace
        if errors:
            output["harness_errors"] = errors
        return output


class Standardizer:
    """High-level Python API for standardizing Norwegian text with NormForm."""

    def __init__(
        self,
        *,
        lang: str = "nob",
        style: str | Path | None = None,
        preserve_unreviewed: bool = False,
        config_path: str | Path | None = None,
        model_identifier: str | None = None,
        cache_path: str | Path | None = None,
        allow_model_mismatch: bool = False,
        debug: bool = False,
        device: str = "auto",
    ):
        if lang not in DEFAULT_MODEL_BY_LANG:
            raise ValueError(
                f"Unsupported language '{lang}'. Supported values are 'nob' and 'nno'."
            )
        if model_identifier is None:
            model_identifier = DEFAULT_MODEL_BY_LANG[lang]
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Model configuration file not found: {config_path}"
                )
        self.config_path = config_path
        self.lang = lang
        self.model_identifier = model_identifier
        self.allow_model_mismatch = allow_model_mismatch
        self.debug = debug
        self.device = device
        self.style_profile = load_style_profile(style, lang=lang) if style else None
        self.preserve_unreviewed = preserve_unreviewed
        self.adapter, self.config = load_adapter(
            config_path,
            model_identifier,
            device_name=device,
            require_multitask=True,
        )
        self.identifier = (
            f"{self.config['model']['id']}@{self.config['model']['revision']}:"
            f"{pinned_model_identifier(model_identifier)}"
        )

        if cache_path is not None:
            self.persistent_cache = True
            self.cache_path = Path(cache_path)
            self.cache = ChoiceCache(
                self.cache_path,
                model_identifier=self.identifier,
                allow_model_mismatch=self.allow_model_mismatch,
            )
        else:
            self.persistent_cache = False
            self._temp_dir = tempfile.TemporaryDirectory(prefix="normform_cache_")
            self.cache_path = Path(self._temp_dir.name) / "style-cache-ephemeral.json"
            self.cache = ChoiceCache(self.cache_path, model_identifier=self.identifier)

    def __call__(
        self,
        text: str,
        prompt_words: Sequence[str] | str | None = None,
        *,
        include_trace: bool = False,
    ) -> dict[str, Any] | str:
        return self.standardize(
            text, prompt_words=prompt_words, include_trace=include_trace
        )

    def standardize(
        self,
        text: str,
        prompt_words: Sequence[str] | str | None = None,
        include_trace: bool = False,
        text_field: str = "text",
    ) -> dict[str, Any] | str:
        harness = DocumentHarness(
            self.adapter,
            self.cache,
            style_profile=self.style_profile,
            preserve_unreviewed=self.preserve_unreviewed,
            debug=self.debug,
        )
        words = parse_prompt_words(prompt_words)
        result = harness.process_record(
            {text_field: text},
            record_index=0,
            default_prompt_words=words,
            include_trace=include_trace,
            text_field=text_field,
        )
        if include_trace:
            return result
        out_field = f"standardised_{text_field}"
        return str(result[out_field])

    def alternatives(self, text: str) -> str:
        """Expand text into [alt1|alt2] morphological alternative brackets."""
        spans = split_sentences(text)
        parts: list[str] = []
        for span in spans:
            if not span.text.strip():
                parts.append(span.text)
                continue
            leading, core, trailing = _strip_ws(span.text)
            res = self.adapter.alternatives(core)
            parts.append(leading + str(res.get("text", core)) + trailing)
        return "".join(parts)


__all__ = [
    "CACHE_SCHEMA",
    "ChoiceCache",
    "DEFAULT_MODEL_BY_LANG",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NNO_MODEL_ID",
    "DEFAULT_NNO_MODEL_REVISION",
    "DEFAULT_NOB_MODEL_ID",
    "DEFAULT_NOB_MODEL_REVISION",
    "DocumentHarness",
    "HarnessSummary",
    "SentenceSpan",
    "Standardizer",
    "choice_key",
    "forced_choice",
    "model_revision",
    "parse_prompt_words",
    "pinned_model_identifier",
    "split_sentences",
]

"""Style cache management, YAML safe-loading for Norwegian words, and Hugging Face Hub integration."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence

import yaml

CACHE_SCHEMA = 1

DEFAULT_STYLE_REPOS = {
    "nob": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "nno": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
}

DEFAULT_STYLE_REVISIONS = {
    "NbAiLab/normform-bokmaal-redaksjonsnorm": "8542693a8e5fe5e7294e0cee1c87019344bec03d",
    "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm": "ceaa918476e2ffcba477964922b580c1c19feb71",
}

STYLE_ALIASES = {
    "redaksjonsnorm": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "redaksjonsstil": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "bokmaal-redaksjonsnorm": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "bokmaal": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "nob": "NbAiLab/normform-bokmaal-redaksjonsnorm",
    "nasjonalbiblioteknorm": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
    "nasjonalbibliotekstil": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
    "nynorsk-nasjonalbiblioteknorm": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
    "nynorsk": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
    "nno": "NbAiLab/normform-nynorsk-nasjonalbiblioteknorm",
}


class NormFormYamlLoader(yaml.SafeLoader):
    """Preserve Norwegian words such as 'no'/'NO' as strings while accepting true/false booleans."""


def _construct_normform_bool(
    loader: NormFormYamlLoader, node: yaml.nodes.ScalarNode
) -> bool | str:
    value = loader.construct_scalar(node)
    if value.casefold() == "true":
        return True
    if value.casefold() == "false":
        return False
    return value


NormFormYamlLoader.add_constructor("tag:yaml.org,2002:bool", _construct_normform_bool)


def load_yaml(text: str) -> Any:
    """Parse YAML safely without turning Norwegian words like 'no' into booleans."""
    return yaml.load(text, Loader=NormFormYamlLoader)


def yaml_choice_scalar(value: str) -> str:
    """Quote words that standard YAML parsers might otherwise interpret as non-strings."""
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = None
    return (
        value
        if isinstance(parsed, str) and parsed == value
        else json.dumps(value, ensure_ascii=False)
    )


def choice_key(candidates: Sequence[str]) -> str:
    return "\x1f".join(candidates)


def canonical_choice_key(candidates: Sequence[str]) -> str:
    """Use one casing- and order-independent key for each alternative set."""
    return "\x1f".join(sorted({candidate.strip().lower() for candidate in candidates}))


def invalid_pair_signature(candidates: Sequence[str]) -> tuple[str, ...]:
    """Match an exact alternative set, independent of case and order."""
    return tuple(sorted({candidate.strip().lower() for candidate in candidates}))


def is_invalid_pair(
    candidates: Sequence[str], invalid_pairs: dict[tuple[str, ...], str]
) -> bool:
    return invalid_pair_signature(candidates) in invalid_pairs


def resolve_candidate_choice(
    candidates: Sequence[str],
    original_choice: str,
    cached_choice: str | None,
    invalid_pairs: dict[tuple[str, ...], str],
) -> tuple[str, bool]:
    """Choose a cached form while preserving the source token's casing pattern."""
    if is_invalid_pair(candidates, invalid_pairs):
        return original_choice, False
    if cached_choice is not None:
        target = cached_choice.lower()
        if original_choice.islower():
            cased = target
        elif original_choice.isupper():
            cased = target.upper()
        elif original_choice[:1].isupper() and original_choice[1:].islower():
            cased = target[:1].upper() + target[1:]
        else:
            def segments(value: str) -> tuple[str, ...] | None:
                result = []
                for part in re.split(r"(\W+)", value):
                    if not part:
                        continue
                    if not any(char.isalpha() for char in part):
                        result.append(part)
                    elif part.islower():
                        result.append("lower")
                    elif part.isupper():
                        result.append("upper")
                    else:
                        return None
                return tuple(result)

            pattern = segments(original_choice)
            matches = [
                candidate
                for candidate in candidates
                if candidate.lower() == target
                and pattern is not None
                and segments(candidate) == pattern
            ]
            return (matches[0], True) if len(matches) == 1 else (original_choice, False)
        return (cased, True) if cased in candidates else (original_choice, False)
    return original_choice, False


@dataclass(frozen=True)
class StyleProfile:
    """Loaded style profile containing canonical choices, invalid pairs, and prompt words."""

    source: str
    path: Path
    choices: dict[str, str]
    invalid_pairs: dict[tuple[str, ...], str]
    prompt_words: list[str]
    metadata: dict[str, Any]


def resolve_style_path(
    style: str | Path | None,
    *,
    lang: str = "nob",
    revision: str | None = None,
) -> Path | None:
    """Resolve a style identifier (local path, alias, or HF repo ID) to a local file Path."""
    if style is None:
        return None
    if isinstance(style, Path):
        if not style.is_file():
            raise FileNotFoundError(f"Style cache file not found: {style}")
        return style

    raw = str(style).strip()
    if not raw:
        return None

    local = Path(raw).expanduser()
    if local.is_file():
        return local

    if raw.lower() == "default":
        repo_id = DEFAULT_STYLE_REPOS.get(lang, DEFAULT_STYLE_REPOS["nob"])
    elif raw.lower() in STYLE_ALIASES:
        repo_id = STYLE_ALIASES[raw.lower()]
    else:
        repo_id = raw.removeprefix("hf://")

    filename = "style.yaml"
    if ":" in repo_id:
        repo_id, filename = repo_id.split(":", 1)
    if "@" in repo_id and revision is None:
        repo_id, revision = repo_id.split("@", 1)
    if revision is None:
        revision = DEFAULT_STYLE_REVISIONS.get(repo_id)

    if "/" not in repo_id:
        raise FileNotFoundError(
            f"Style '{style}' is neither an existing local file, a built-in alias "
            f"({', '.join(sorted(STYLE_ALIASES))}), nor a Hugging Face repo ID ('org/name')."
        )

    from huggingface_hub import hf_hub_download

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download style '{repo_id}' ({filename}) from Hugging Face. "
            "Ensure the repository exists and you are logged in (`hf auth login`) if it is private.\n"
            f"Original error: {exc}"
        ) from exc


def load_seed_cache(cache_path: Path | None) -> dict[str, str]:
    """Load human-readable YAML or JSON cache into canonical lowercase dict."""
    if cache_path is None or not cache_path.is_file():
        return {}

    content = cache_path.read_text(encoding="utf-8")
    data = (
        load_yaml(content)
        if cache_path.suffix in (".yaml", ".yml")
        else json.loads(content)
    )
    raw_choices = data.get("choices", data) if isinstance(data, dict) else {}

    canonical: dict[str, str] = {}
    for key, val in raw_choices.items():
        if str(key).startswith("_") or str(key) == "invalid_pairs":
            continue
        if isinstance(val, dict):
            choice = str(val.get("choice", ""))
        else:
            choice = str(val)

        key_str = str(key)
        if " | " in key_str:
            cands = tuple(c.strip() for c in key_str.split(" | "))
        elif "|" in key_str:
            cands = tuple(c.strip() for c in key_str.split("|"))
        elif "\x1f" in key_str:
            cands = tuple(key_str.split("\x1f"))
        else:
            cands = tuple(key_str.split(","))

        canonical_key = canonical_choice_key(cands)
        normalized_choice = choice.lower()
        if normalized_choice not in {candidate.lower() for candidate in cands}:
            raise ValueError(f"Cache choice {choice!r} is outside {key!r}")
        previous = canonical.get(canonical_key)
        if previous is not None and previous != normalized_choice:
            raise ValueError(f"Conflicting case variants for cache key {key!r}")
        canonical[canonical_key] = normalized_choice
    return canonical


def load_invalid_pairs(cache_path: Path | None) -> dict[tuple[str, ...], str]:
    """Load entries that preserve the source token instead of choosing a form."""
    if cache_path is None or not cache_path.is_file():
        return {}

    content = cache_path.read_text(encoding="utf-8")
    data = (
        load_yaml(content)
        if cache_path.suffix in (".yaml", ".yml")
        else json.loads(content)
    )
    if not isinstance(data, dict):
        return {}
    raw_pairs = data.get("invalid_pairs", {})
    if not isinstance(raw_pairs, dict):
        raise ValueError("invalid_pairs must be a mapping from candidate keys to reasons")
    invalid: dict[tuple[str, ...], str] = {}
    for key, reason in raw_pairs.items():
        candidates = [
            part.strip() for part in str(key).replace("\x1f", "|").split("|")
        ]
        if len(candidates) < 2 or any(not candidate for candidate in candidates):
            raise ValueError(f"Invalid candidate pair: {key!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Invalid pair needs a reason: {key!r}")
        signature = invalid_pair_signature(candidates)
        if signature in invalid:
            raise ValueError(f"Duplicate invalid candidate pair: {key!r}")
        invalid[signature] = reason.strip()
    return invalid


def load_style_profile(
    style: str | Path | None,
    *,
    lang: str = "nob",
    revision: str | None = None,
) -> StyleProfile | None:
    """Resolve and load a complete StyleProfile from local path, alias, or HF repo."""
    path = resolve_style_path(style, lang=lang, revision=revision)
    if path is None:
        return None
    content = path.read_text(encoding="utf-8")
    data = load_yaml(content) if path.suffix in (".yaml", ".yml") else json.loads(content)
    metadata = data.get("_metadata", {}) if isinstance(data, dict) else {}
    invalid_pairs = load_invalid_pairs(path)
    choices = {
        key: choice
        for key, choice in load_seed_cache(path).items()
        if not is_invalid_pair(key.split("\x1f"), invalid_pairs)
    }
    raw_words = metadata.get("prompt_words", [])
    if isinstance(raw_words, str):
        prompt_words = [w.strip() for w in raw_words.split(",") if w.strip()]
    elif isinstance(raw_words, list):
        prompt_words = [str(w).strip() for w in raw_words if str(w).strip()]
    else:
        prompt_words = []
    return StyleProfile(
        source=str(style),
        path=path,
        choices=choices,
        invalid_pairs=invalid_pairs,
        prompt_words=prompt_words,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def pull_style(
    style: str,
    output_path: Path,
    *,
    lang: str = "nob",
    revision: str | None = None,
) -> Path:
    """Download a style cache from Hugging Face (or alias) to a local file."""
    resolved = resolve_style_path(style, lang=lang, revision=revision)
    if resolved is None:
        raise ValueError("Style identifier is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, output_path)
    return output_path


def push_style(
    style_path: Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Publish NormForm style cache",
) -> str:
    """Publish a local YAML style cache to a Hugging Face dataset repository."""
    if not style_path.is_file():
        raise FileNotFoundError(f"Style file not found: {style_path}")
    # Validate before uploading
    profile = load_style_profile(style_path)
    if profile is None:
        raise ValueError(f"Could not parse style file: {style_path}")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    sha256 = hashlib.sha256(style_path.read_bytes()).hexdigest()
    lang = profile.metadata.get("language", "nob")
    words_str = ", ".join(profile.prompt_words) if profile.prompt_words else "(none)"
    readme = f"""---
language:
- {lang}
- no
license: apache-2.0
tags:
- normform
- morphology
- norwegian
- style-cache
---

# NormForm Style: `{repo_id}`

- **Language**: `{lang}`
- **File**: `style.yaml`
- **SHA-256**: `{sha256}`
- **Total Choices**: {len(profile.choices):,}
- **Invalid Pairs (Preserved Unchanged)**: {len(profile.invalid_pairs):,}
- **Steering Prompt Words**: `{words_str}`

## Usage

```bash
normform standardize input.txt --lang {lang} --style {repo_id}
```
"""
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        operations=[
            CommitOperationAdd(path_in_repo="style.yaml", path_or_fileobj=str(style_path)),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")),
        ],
    )
    return str(getattr(commit, "oid", commit.commit_url))


def _model_fingerprint(identifier: str | None) -> str | None:
    if not identifier:
        return None
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


class ChoiceCache:
    """An explicit document-style scope persisted as one atomic JSON file."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_identifier: str | None = None,
        allow_model_mismatch: bool = False,
    ):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.model_identifier = model_identifier
        self.allow_model_mismatch = allow_model_mismatch
        self.data = self._read()
        self._check_model()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": CACHE_SCHEMA,
                "scope": "document",
                "choices": {},
                "model_fingerprint": _model_fingerprint(self.model_identifier),
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != CACHE_SCHEMA
            or payload.get("scope") != "document"
        ):
            raise ValueError("Unsupported or non-document choice cache")
        if not isinstance(payload.get("choices"), dict):
            raise ValueError("Choice cache choices must be an object")
        return payload

    def _check_model(self) -> None:
        expected = _model_fingerprint(self.model_identifier)
        existing = self.data.get("model_fingerprint")
        if (
            expected
            and existing
            and expected != existing
            and not self.allow_model_mismatch
        ):
            raise ValueError(
                "Choice cache belongs to a different model; pass --allow-model-mismatch"
            )
        if expected and not existing:
            self.data["model_fingerprint"] = expected

    def get(self, candidates: Sequence[str]) -> str | None:
        item = self.data["choices"].get(choice_key(candidates))
        return None if item is None else str(item["choice"])

    def clear(self) -> None:
        self.data["choices"].clear()

    def put(
        self,
        candidates: Sequence[str],
        choice: str,
        *,
        record_index: int,
        sentence_index: int,
        force: bool = False,
    ) -> None:
        candidates = tuple(candidates)
        if choice not in candidates:
            raise ValueError("Cached choice is outside its alternative group")
        key = choice_key(candidates)
        old = self.data["choices"].get(key)
        if old is not None and old["choice"] != choice and not force:
            raise ValueError("A document cache cannot change an existing choice")
        if old is None or old["choice"] != choice:
            self.data["choices"][key] = {
                "alternatives": list(candidates),
                "choice": choice,
                "first_record": record_index,
                "first_sentence": sentence_index,
            }
            if old is not None:
                self.data["choices"][key]["overrode_choice"] = old["choice"]

    @contextmanager
    def transaction(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                self.data = self._read()
                self._check_model()
            yield self
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=self.path.name + ".", dir=self.path.parent
            )
            try:
                with open(fd, "w", encoding="utf-8", closefd=True) as handle:
                    json.dump(
                        self.data, handle, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    handle.write("\n")
                Path(temporary).replace(self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

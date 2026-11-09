"""Model adapter for valid alternatives and structurally constrained style output."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .contract import DIRECT_TASK, STYLE_RULE, STYLE_TASK, clean_direct, parse
from .styles import load_yaml

DEFAULT_NOB_MODEL_ID = "NbAiLab/normform-bokmaal-multitask"
DEFAULT_NOB_MODEL_REVISION = "eb83a85c52bea137e2bbb3ba95e29be3e9476923"
DEFAULT_NNO_MODEL_ID = "NbAiLab/normform-nynorsk-multitask"
DEFAULT_NNO_MODEL_REVISION = "350cf9d3aeeba2f390353f4147ef2910dff7b0ee"
DEFAULT_MODEL_BY_LANG = {
    "nob": DEFAULT_NOB_MODEL_ID,
    "nno": DEFAULT_NNO_MODEL_ID,
}
DEFAULT_MODEL_ID = DEFAULT_NOB_MODEL_ID

DEFAULT_BASE_MODEL_ID = "google/t5gemma-2-1b-1b"
DEFAULT_BASE_MODEL_REVISION = "dd0a2683227859151b1730ca3a63087df5b5f39b"

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def default_runtime_config() -> dict[str, Any]:
    """Built-in runtime model configuration, requiring no external YAML files on disk."""
    return {
        "model": {
            "id": DEFAULT_BASE_MODEL_ID,
            "revision": DEFAULT_BASE_MODEL_REVISION,
            "dtype": "bfloat16",
            "master_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "sdpa_backend": "math",
            "freeze_vision": True,
            "gradient_checkpointing": False,
        },
        "data": {
            "max_source_length": 1024,
            "max_target_length": 768,
        },
    }


def model_revision(identifier: str | Path) -> str | None:
    if str(identifier) == DEFAULT_NOB_MODEL_ID:
        return DEFAULT_NOB_MODEL_REVISION
    if str(identifier) == DEFAULT_NNO_MODEL_ID:
        return DEFAULT_NNO_MODEL_REVISION
    return None


def pinned_model_identifier(identifier: str | Path) -> str:
    revision = model_revision(identifier)
    return f"{identifier}@{revision}" if revision else str(identifier)


@dataclass(frozen=True)
class Resolution:
    text: str
    choices: tuple[str, ...]
    used_constrained_search: bool
    candidates_scored: int


def resolve(
    input_text: str,
    proposal: str | None,
    score: Callable[[Sequence[str]], Sequence[float]],
    beam_size: int = 4,
) -> Resolution:
    """Constrained style resolution whose return path always renders locked literals."""
    if beam_size < 1:
        raise ValueError("Positive beam size required")
    template = parse(input_text)
    if not template.groups:
        return Resolution(template.render(()), (), False, 0)
    if proposal is not None:
        try:
            choices = template.align(proposal)
        except ValueError:
            pass
        else:
            return Resolution(template.render(choices), choices, False, 0)
    beam = [tuple(group[0] for group in template.groups)]
    scored = 0
    for index, group in enumerate(template.groups):
        candidates = sorted(
            {
                choices[:index] + (candidate,) + choices[index + 1 :]
                for choices in beam
                for candidate in group
            }
        )
        renderings = [template.render(choices) for choices in candidates]
        values = list(score(renderings))
        if len(values) != len(candidates):
            raise ValueError("Scorer returned incorrect candidate count")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Nonfinite candidate score")
        scored += len(candidates)
        ordered = sorted(zip(values, candidates), key=lambda item: (-item[0], item[1]))
        beam = [choices for _, choices in ordered[:beam_size]]
    selected = beam[0]
    return Resolution(template.render(selected), selected, True, scored)


def target_token_ids(
    tokenizer: Any,
    targets: list[str],
    max_length: int | None = None,
) -> list[list[int]]:
    """Tokenize decoder targets and guarantee exactly one terminal EOS token."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Seq2seq tokenizer must define eos_token_id")
    if max_length is not None and max_length < 2:
        raise ValueError("Target max_length must leave room for content and EOS")
    encoded = tokenizer(
        text_target=targets,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    result: list[list[int]] = []
    for values in encoded:
        ids = list(values)
        if not ids or ids[-1] != eos_token_id:
            ids.append(eos_token_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = eos_token_id
        result.append(ids)
    return result


def normalize_direct_proposal(source: str, proposal: str) -> str:
    """Repair only redundant brackets and boundary whitespace, then validate normally."""
    def collapse(match: re.Match[str]) -> str:
        parts = match.group(1).split("|")
        candidate = parts[0].strip()
        if len(parts) > 1 and candidate and all(part.strip() == candidate for part in parts):
            return candidate
        return match.group(0)

    proposal = re.sub(r"\[([^\[\]]+)\]", collapse, proposal)
    leading = source[: len(source) - len(source.lstrip())]
    trailing = source[len(source.rstrip()) :]
    return leading + proposal.strip() + trailing


class MultitaskModel:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_source_length: int = 1024,
        max_target_length: int = 768,
        score_batch_size: int = 4,
        beam_size: int = 4,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.score_batch_size = score_batch_size
        self.beam_size = beam_size
        if min(max_source_length, max_target_length, score_batch_size, beam_size) < 1:
            raise ValueError("Positive token, batch, and beam limits required")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def precision(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts), padding=True, truncation=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > self.max_source_length:
            raise ValueError("Prompt exceeds model input capacity; refusing truncation")
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.inference_mode()
    def propose_batch(
        self, prompts: Sequence[str], *, temperature: float = 0.0
    ) -> list[str]:
        if not prompts:
            return []
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        inputs = self.encode(list(prompts))
        generation_kwargs: dict[str, Any] = {
            "max_length": self.max_target_length,
            "num_beams": 1,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        with self.precision():
            generated = self.model.generate(**inputs, **generation_kwargs)
        raw_list = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results = []
        for prompt, raw in zip(prompts, raw_list):
            if raw.startswith(prompt):
                raw = raw[len(prompt) :].lstrip("\n ")
            for prefix in (f"{DIRECT_TASK}:\ntekst: ", f"{DIRECT_TASK}:\ntekst:"):
                if raw.startswith(prefix):
                    raw = raw[len(prefix) :].lstrip()
            if raw.startswith(f"{STYLE_TASK}:"):
                if "\nsetning: " in raw:
                    raw = raw.split("\nsetning: ", 1)[1]
            results.append(raw)
        return results

    def propose(self, prompt: str, *, temperature: float = 0.0) -> str:
        return self.propose_batch([prompt], temperature=temperature)[0]

    @torch.inference_mode()
    def score(self, prompt: str, candidates: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(candidates), self.score_batch_size):
            batch = list(candidates[start : start + self.score_batch_size])
            labels = target_token_ids(self.tokenizer, batch)
            if any(len(ids) > self.max_target_length for ids in labels):
                raise ValueError(
                    "Candidate exceeds model target capacity; refusing truncation"
                )
            targets = torch.full(
                (len(batch), max(map(len, labels))),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            for i, ids in enumerate(labels):
                targets[i, : len(ids)] = torch.tensor(ids, device=self.device)
            inputs = self.encode([prompt] * len(batch))
            with self.precision():
                output = self.model(**inputs, labels=targets, use_cache=False)
            for i, ids in enumerate(labels):
                total = 0.0
                for offset in range(0, len(ids), 64):
                    logits = output.logits[
                        i, offset : offset + min(64, len(ids) - offset)
                    ].float()
                    chosen = targets[i, offset : offset + len(logits)]
                    logp = logits.gather(-1, chosen.unsqueeze(-1)).squeeze(
                        -1
                    ) - torch.logsumexp(logits, dim=-1)
                    total += float(logp.sum().item())
                scores.append(total)
            del output, targets, inputs
        return scores

    @torch.inference_mode()
    def score_pairs(
        self, prompts: Sequence[str], targets: Sequence[str]
    ) -> list[float]:
        if len(prompts) != len(targets):
            raise ValueError("Prompts and targets must have same length")
        scores: list[float] = []
        for start in range(0, len(prompts), self.score_batch_size):
            p_batch = list(prompts[start : start + self.score_batch_size])
            t_batch = list(targets[start : start + self.score_batch_size])
            labels = target_token_ids(self.tokenizer, t_batch)
            if any(len(ids) > self.max_target_length for ids in labels):
                raise ValueError(
                    "Candidate exceeds model target capacity; refusing truncation"
                )
            t_tensor = torch.full(
                (len(t_batch), max(map(len, labels))),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            for i, ids in enumerate(labels):
                t_tensor[i, : len(ids)] = torch.tensor(ids, device=self.device)
            inputs = self.encode(p_batch)
            with self.precision():
                output = self.model(**inputs, labels=t_tensor, use_cache=False)
            for i, ids in enumerate(labels):
                total = 0.0
                for offset in range(0, len(ids), 64):
                    logits = output.logits[
                        i, offset : offset + min(64, len(ids) - offset)
                    ].float()
                    chosen = t_tensor[i, offset : offset + len(logits)]
                    logp = logits.gather(-1, chosen.unsqueeze(-1)).squeeze(
                        -1
                    ) - torch.logsumexp(logits, dim=-1)
                    total += float(logp.sum().item())
                scores.append(total)
            del output, t_tensor, inputs
        return scores

    def alternatives_batch(
        self, texts: Sequence[str], *, temperature: float = 0.0
    ) -> list[str]:
        if not texts:
            return []
        prompts = [f"{DIRECT_TASK}:\ntekst: {t}" for t in texts]
        proposals = self.propose_batch(prompts, temperature=temperature)
        outputs = []
        for text, proposal in zip(texts, proposals):
            try:
                out = clean_direct(text, normalize_direct_proposal(text, proposal))
                parse(out).align(text)
                outputs.append(out)
            except Exception:
                outputs.append(text)
        return outputs

    def alternatives(self, text: str, *, temperature: float = 0.0) -> dict[str, Any]:
        source = parse(text)
        if source.groups:
            raise ValueError(
                "Direct task requires ordinary text, not supplied alternative groups"
            )
        prompt = f"{DIRECT_TASK}:\ntekst: {text}"
        proposal = self.propose(prompt, temperature=temperature)
        normalized_proposal = normalize_direct_proposal(text, proposal)
        try:
            output = clean_direct(text, normalized_proposal)
            fallback = False
        except ValueError:
            output = text
            fallback = True
        parse(output).align(text)
        return {
            "task": DIRECT_TASK,
            "text": output,
            "raw_proposal": proposal,
            "source_copy_fallback": fallback,
            "prohibited_family_cleanup": not fallback and output != normalized_proposal,
        }

    def style_batch(
        self,
        evidence: Sequence[str],
        input_texts: Sequence[str],
        *,
        temperature: float = 0.0,
        batch_size: int = 256,
    ) -> list[dict[str, Any]]:
        """Vectorized style resolution: proposals generated in GPU batches, aligned structurally."""
        if not input_texts:
            return []
        prefix = (
            f"{STYLE_TASK}:\n{STYLE_RULE}\nmorfologiske_stilvalg: "
            + " | ".join(evidence)
            + "\nsetning: "
        )
        prompts = [f"{prefix}{t}" for t in input_texts]

        all_proposals: list[str] = []
        for start in range(0, len(prompts), batch_size):
            slice_prompts = prompts[start : start + batch_size]
            slice_proposals = self.propose_batch(slice_prompts, temperature=temperature)
            all_proposals.extend(slice_proposals)

        results: list[dict[str, Any]] = []
        for input_text, prompt, proposal in zip(input_texts, prompts, all_proposals):
            template = parse(input_text)
            if not template.groups:
                results.append(
                    {
                        "task": STYLE_TASK,
                        "text": input_text,
                        "raw_proposal": proposal,
                        "choices": (),
                    }
                )
                continue
            try:
                choices = template.align(proposal)
                results.append(
                    {
                        "task": STYLE_TASK,
                        "text": template.render(choices),
                        "raw_proposal": proposal,
                        "choices": choices,
                        "used_constrained_search": False,
                        "candidates_scored": 0,
                    }
                )
            except ValueError:
                try:
                    res = resolve(
                        input_text,
                        proposal,
                        lambda candidates: self.score(prompt, candidates),
                        self.beam_size,
                    )
                    results.append(
                        {"task": STYLE_TASK, "raw_proposal": proposal, **asdict(res)}
                    )
                except Exception:
                    fallback_choices = tuple(grp[0] for grp in template.groups)
                    results.append(
                        {
                            "task": STYLE_TASK,
                            "text": template.render(fallback_choices),
                            "raw_proposal": proposal,
                            "choices": fallback_choices,
                            "used_constrained_search": False,
                            "candidates_scored": 0,
                        }
                    )
        return results

    def style(
        self,
        evidence: Sequence[str],
        input_text: str,
        *,
        temperature: float = 0.0,
        strict: bool = False,
    ) -> dict[str, Any]:
        template = parse(input_text)
        prompt = (
            f"{STYLE_TASK}:\n{STYLE_RULE}\nmorfologiske_stilvalg: "
            + " | ".join(evidence)
            + "\nsetning: "
            + input_text
        )
        proposal = (
            self.propose(prompt, temperature=temperature) if template.groups else None
        )
        if strict and proposal is not None:
            try:
                template.align(proposal)
            except ValueError:
                return {
                    "task": STYLE_TASK,
                    "raw_proposal": proposal,
                    "text": input_text,
                    "invalid_proposal": True,
                    "used_constrained_search": False,
                    "candidates_scored": 0,
                }
        result = resolve(
            input_text,
            proposal,
            lambda candidates: self.score(prompt, candidates),
            self.beam_size,
        )
        return {"task": STYLE_TASK, "raw_proposal": proposal, **asdict(result)}

    def pipeline(self, text: str, evidence: Sequence[str]) -> dict[str, Any]:
        alternatives = self.alternatives(text)
        styled = self.style(evidence, alternatives["text"])
        return {"input_text": text, "alternatives": alternatives, "style": styled}


def _verify_harness_checkpoint(checkpoint_path: Path, identifier: str | Path) -> None:
    """Reject default releases that do not declare both harness task contracts."""
    if str(identifier) not in DEFAULT_MODEL_BY_LANG.values():
        return
    provenance_path = checkpoint_path / "normform_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            f"The published model {identifier} lacks a verified two-task NormForm release. "
            "It cannot safely serve the harness's alternatives and style prompts. "
            "Use a verified multitask checkpoint with --model."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {DIRECT_TASK, STYLE_TASK}
    if (
        provenance.get("model") != str(identifier)
        or set(provenance.get("tasks", ())) != expected
    ):
        raise RuntimeError(
            f"The published model {identifier} does not declare both NormForm tasks"
        )


def _load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = config["model"]
    reference = str(model_cfg["pretrained_path"])
    sdpa_backend = str(model_cfg.get("sdpa_backend", "auto")).lower()
    if sdpa_backend == "math" and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    master_dtype_name = str(model_cfg.get("master_dtype", "bfloat16")).lower()
    dtype = DTYPES.get(master_dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(reference)
    if tokenizer.pad_token_id is None:
        raise ValueError("Seq2seq tokenizer must define pad_token_id")
    if tokenizer.eos_token_id is None:
        raise ValueError("Seq2seq tokenizer must define eos_token_id")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        reference,
        dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    if model_cfg.get("freeze_vision", True):
        for name, parameter in model.named_parameters():
            if any(component in name for component in ("vision_tower", "vision_model")):
                parameter.requires_grad = False
    decoder = getattr(model.config, "decoder", None)
    if decoder is not None:
        decoder.use_cache = False
    else:
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    return model, tokenizer


def load_adapter(
    config_path: Path | str | None,
    checkpoint: Path | str,
    device_name: str = "auto",
    *,
    require_multitask: bool = False,
) -> tuple[MultitaskModel, dict[str, Any]]:
    """Load the two-task neural model adapter using built-in defaults or an optional config YAML."""
    if not device_name or device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"[NormForm Error] Requested device '{device_name}', but PyTorch CUDA support is not available."
        )
    else:
        device = torch.device(device_name)

    checkpoint_path = Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_dir():
        from huggingface_hub import snapshot_download

        try:
            checkpoint_path = Path(
                snapshot_download(
                    repo_id=str(checkpoint),
                    repo_type="model",
                    revision=model_revision(checkpoint),
                )
            )
        except Exception as exc:
            msg = (
                f"\n\n[NormForm Error] Failed to access or download model '{checkpoint}'.\n"
                "If this is a private or gated Hugging Face repository (such as 'NbAiLab/normform-bokmaal-multitask'), "
                "you need to authenticate with Hugging Face:\n\n"
                "  1. Run in your terminal:  hf auth login   (or: huggingface-cli login)\n"
                "  2. Or set environment variable:  export HF_TOKEN=\"hf_...\"\n"
                "  3. Alternatively, pass a local model directory:  --model /path/to/local/checkpoint\n\n"
                f"Original error: {exc}\n"
            )
            raise RuntimeError(msg) from exc

    if require_multitask:
        _verify_harness_checkpoint(checkpoint_path, checkpoint)

    config = default_runtime_config()
    if config_path is not None and Path(config_path).is_file():
        loaded = load_yaml(Path(config_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            if isinstance(loaded.get("model"), dict):
                config["model"].update(loaded["model"])
            if isinstance(loaded.get("data"), dict):
                config["data"].update(loaded["data"])

    config["model"]["pretrained_path"] = str(checkpoint_path)
    config["model"]["master_dtype"] = "bfloat16"
    model, tokenizer = _load_model_and_tokenizer(config)
    model = model.to(device)
    return (
        MultitaskModel(
            model,
            tokenizer,
            max_source_length=int(config["data"]["max_source_length"]),
            max_target_length=int(config["data"]["max_target_length"]),
        ),
        config,
    )

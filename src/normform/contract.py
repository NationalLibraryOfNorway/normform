"""Strict, dependency-free contracts for the prompt-conditioned multitask lattice."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

DIRECT_TASK = "create_morphological_alternatives"
STYLE_TASK = "skriv_i_samme_morfologiske_stil"
STYLE_RULE = "bruk bare valgene i hakeparenteser; behold annen tekst uendret."
PROHIBITED = {
    frozenset(("et", "ett")),
    frozenset(("en", "én")),
    frozenset(("eit", "eitt")),
    frozenset(("ein", "éin")),
    frozenset(("og", "òg")),
    frozenset(("ogso", "også")),
}


@dataclass(frozen=True)
class Template:
    literals: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]

    def render(self, choices: tuple[str, ...]) -> str:
        if len(choices) != len(self.groups):
            raise ValueError("Wrong number of choices")
        parts = [self.literals[0]]
        for index, (choice, candidates) in enumerate(zip(choices, self.groups)):
            if choice not in candidates:
                raise ValueError("Choice outside supplied candidates")
            parts.extend((choice, self.literals[index + 1]))
        return "".join(parts)

    def align(self, text: str) -> tuple[str, ...]:
        """Find exact rendering without greedily confusing prefix alternatives."""
        positions: dict[int, tuple[str, ...]] = {0: ()}
        for index, literal in enumerate(self.literals):
            following: dict[int, tuple[str, ...]] = {}
            for offset, choices in positions.items():
                if not text.startswith(literal, offset):
                    continue
                offset += len(literal)
                if index == len(self.groups):
                    following[offset] = choices
                else:
                    for candidate in self.groups[index]:
                        if text.startswith(candidate, offset):
                            following.setdefault(
                                offset + len(candidate), choices + (candidate,)
                            )
            positions = following
        if len(text) not in positions:
            raise ValueError("Text changes a locked literal or uses an invalid choice")
        return positions[len(text)]


def parse(text: str) -> Template:
    """Parse bracketed alternative groups while treating plain [citations] as literals."""
    literals, groups = [], []
    cursor = search = 0
    while True:
        start = text.find("[", search)
        if start < 0:
            tail = text[cursor:]
            if "]" in text[search:]:
                raise ValueError("Unmatched closing bracket")
            literals.append(tail)
            break
        if "]" in text[search:start]:
            raise ValueError("Unmatched closing bracket")
        end = text.find("]", start + 1)
        if end < 0:
            raise ValueError("Unclosed bracket")
        content = text[start + 1 : end]
        if "[" in content:
            raise ValueError("Nested bracket")
        search = end + 1
        if "|" not in content:
            continue
        candidates = tuple(content.split("|"))
        if any(not x for x in candidates):
            raise ValueError("Malformed alternative group")
        if len(set(candidates)) != len(candidates):
            raise ValueError("Duplicate alternatives")
        literals.append(text[cursor:start])
        groups.append(candidates)
        cursor = end + 1
    return Template(tuple(literals), tuple(groups))


def prohibited(candidates: tuple[str, ...]) -> bool:
    return (
        frozenset(unicodedata.normalize("NFC", x).casefold() for x in candidates)
        in PROHIBITED
    )


def clean_direct(source: str, encoded: str) -> str:
    prefix = f"{DIRECT_TASK}:\ntekst: "
    if encoded.startswith(prefix):
        encoded = encoded[len(prefix) :]
    elif encoded.startswith(f"{DIRECT_TASK}:\ntekst:"):
        encoded = encoded[len(f"{DIRECT_TASK}:\ntekst:") :].lstrip()
    template = parse(encoded)
    aligned = template.align(source)
    parts = [template.literals[0]]
    for index, group in enumerate(template.groups):
        parts.extend(
            (
                aligned[index] if prohibited(group) else "[" + "|".join(group) + "]",
                template.literals[index + 1],
            )
        )
    result = "".join(parts)
    cleaned = parse(result)
    cleaned.align(source)
    if any(prohibited(group) for group in cleaned.groups):
        raise ValueError("Residual prohibited group")
    return result


def style_example(
    encoded: str,
    desired: str,
    evidence: tuple[str, ...],
    open_indices: frozenset[int],
    locked: dict[int, str],
) -> dict[str, str]:
    template = parse(encoded)
    selected = template.align(desired)
    indices = set(range(len(template.groups)))
    if not open_indices <= indices or set(locked) != indices - open_indices:
        raise ValueError("Every group must be explicitly open or locked")
    parts, target = [template.literals[0]], []
    for index, group in enumerate(template.groups):
        choice = selected[index] if index in open_indices else locked[index]
        if choice not in group:
            raise ValueError("Invalid locked candidate")
        target.append(choice)
        parts.extend(
            (
                "[" + "|".join(group) + "]" if index in open_indices else choice,
                template.literals[index + 1],
            )
        )
    input_text = "".join(parts)
    output = template.render(tuple(target))
    parse(input_text).align(output)
    return {
        "task": STYLE_TASK,
        "input_text": input_text,
        "target": output,
        "prompt": (
            f"{STYLE_TASK}:\n{STYLE_RULE}\n"
            f"morfologiske_stilvalg: {' | '.join(evidence)}\n"
            f"setning: {input_text}"
        ),
    }

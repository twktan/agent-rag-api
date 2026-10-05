"""Input guardrails: cheap deterministic prompt-injection heuristics.

This is the first of three layers (heuristics -> moderation API -> LLM router's "refuse"
route). Heuristics are fast and explainable but brittle, so they only catch the obvious
attacks; their false-refusal rate on benign questions is measured in eval/run_eval.py.
The same detector screens retrieved chunks to defend against indirect injection.
"""

import re

_PATTERNS: dict[str, str] = {
    "override_instructions":
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|"
        r"your|the|safety)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|guidelines|directives|restrictions)\b",
    "prompt_extraction":
        r"\b(reveal|show|print|repeat|output|display|leak|dump)\b[^.\n]{0,40}\b(system prompt|hidden "
        r"(rules|instructions)|your (instructions|rules|prompt|configuration)|initial prompt)\b",
    "repeat_above": r"\brepeat\b[^.\n]{0,30}\b(text|words|everything|messages?)\b[^.\n]{0,15}\b(above|before)\b",
    "mode_switch": r"\b(developer|dan|jailbreak|god|unrestricted) mode\b",
    "persona_override":
        r"\byou are now\b[^.\n]{0,40}\b(dan|unfiltered|unrestricted|jailbroken|no restrictions|evil)\b",
    "role_spoofing": r"(^|\n)\s*(system|assistant|developer)\s*:",
    "tag_spoofing": r"</?\s*(system|documents?|user_message|instructions)\s*>",
    "secret_request":
        r"\b(reveal|show|print|give|list|leak)\b[^.\n]{0,30}"
        r"\b(api[ _-]?keys?|secret keys?|access tokens?|passwords?)\b",
}
_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in _PATTERNS.items()}


def detect_injection(text: str) -> list[str]:
    """Names of the injection heuristics that match `text` (empty list if none)."""
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


def filter_injected_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split retrieved chunks into (clean, suspicious) to block indirect prompt injection."""
    clean, suspicious = [], []
    for chunk in chunks:
        (suspicious if detect_injection(chunk["text"]) else clean).append(chunk)
    return clean, suspicious

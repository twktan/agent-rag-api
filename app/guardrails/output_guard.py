"""Output guardrails: system-prompt leak detection (canary) and PII redaction."""

import re

_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net", "domain.com", "email.com")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b")  # Singapore NRIC/FIN
_PHONE = re.compile(r"(?<![\w+])\+\d{1,3}[\s-]\d{3,4}[\s-]?\d{4}\b")  # international format only
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


def leaked_canary(text: str, canary: str) -> bool:
    return canary in text


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Redact personal data from model output. Returns (text, kinds_redacted)."""
    found: list[str] = []

    def sub_email(m: re.Match) -> str:
        if m.group(1).lower() in _EXAMPLE_DOMAINS:  # keep illustrative addresses in code answers
            return m.group(0)
        found.append("email")
        return "[REDACTED_EMAIL]"

    def sub_card(m: re.Match) -> str:
        if not _luhn_ok(m.group(0)):
            return m.group(0)
        found.append("card")
        return "[REDACTED_CARD]"

    text = _EMAIL.sub(sub_email, text)
    text, n = _NRIC.subn("[REDACTED_NRIC]", text)
    found += ["nric"] * n
    text, n = _PHONE.subn("[REDACTED_PHONE]", text)
    found += ["phone"] * n
    text = _CARD.sub(sub_card, text)
    return text, sorted(set(found))

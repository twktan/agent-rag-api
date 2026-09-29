"""Prompt helpers: XML-style delimiting of untrusted text and shared canned responses."""

import re

ABSTAIN_MESSAGE = ("I couldn't find this in TTI's internal documentation. Please check with the "
                   "relevant team (for example HR, IT or your manager).")
REFUSAL_MESSAGE = ("I can't help with that request. I can answer questions about TTI's policies, "
                   "tools and products, or help with general work tasks.")

_TAGS = "documents?|question|user_message|previous_answer|reviewer_feedback|system"
_TAG_RE = re.compile(rf"</?\s*({_TAGS})\b[^>]*>", re.IGNORECASE)


def neutralize(text: str) -> str:
    """Strip delimiter-like tags from untrusted text so it cannot close or spoof our sections."""
    return _TAG_RE.sub("", text)


def tag(name: str, text: str, **attrs: str) -> str:
    attr_str = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f"<{name}{attr_str}>\n{neutralize(text)}\n</{name}>"


def is_abstention(answer: str) -> bool:
    return answer.strip().startswith(ABSTAIN_MESSAGE[:40])

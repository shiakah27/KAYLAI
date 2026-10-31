"""Parse a model reply: the strict FINAL line, the ANSWER line and the THINKING section."""

import re

# Strict on purpose: the line must start with exactly "FINAL:". "**FINAL:**" or "Final answer:"
# do not count, because a reply that breaks the format is a useful signal for verification.
FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.MULTILINE)
ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.MULTILINE)
THINKING_RE = re.compile(r"THINKING:\s*(.*?)(?=^\s*ANSWER:|^\s*FINAL:|\Z)", re.DOTALL | re.MULTILINE)


def parse_final(text: str | None) -> str | None:
    """The value of the last FINAL line, or None."""
    if not text:
        return None
    matches = FINAL_RE.findall(text)
    return matches[-1] if matches else None


def parse_answer(text: str | None) -> str | None:
    """The value of the last ANSWER line (the Arabic sentence for the user), or None."""
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def parse_thinking(text: str | None) -> str | None:
    """The THINKING section, for models without a separate reasoning channel, or None."""
    if not text:
        return None
    m = THINKING_RE.search(text)
    return m.group(1).strip() if m and m.group(1).strip() else None

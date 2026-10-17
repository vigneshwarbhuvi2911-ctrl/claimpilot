"""Guardrails around every LLM call.

Input: personal data is replaced by placeholders before any document text reaches the model
(known customer details from the policy record, labelled name fields, and generic patterns for
phone numbers, emails, Aadhaar, PAN and licence numbers). The mapping stays in memory so the
handler-facing summary can be shown with real values.

Output: the model's evidence quotes must appear in the source documents (grounding check), and
the recommendation it writes may not change the decision or amounts computed by the rules engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"(?<!\w)\+?91[\s-]?\d{5}[\s-]?\d{5}(?!\w)|(?<!\w)[6-9]\d{9}(?!\w)")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("LICENCE_NO", re.compile(r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b")),
]
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
NAME_LABEL = re.compile(r"(?im)^(\s*(?:insured name|patient name|name|policyholder)\s*:\s*)(.+)$")


@dataclass
class Masker:
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original
    _reverse: dict[str, str] = field(default_factory=dict)  # original -> placeholder
    _counts: dict[str, int] = field(default_factory=dict)

    def _token(self, kind: str, value: str) -> str:
        value = value.strip()
        if value in self._reverse:
            return self._reverse[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        tok = f"[{kind}_{self._counts[kind]}]"
        self.mapping[tok], self._reverse[value] = value, tok
        return tok

    def add_known(self, kind: str, value: str | None) -> None:
        if value and value.strip():
            self._token(kind, value)

    def mask(self, text: str) -> str:
        # Known values first (longest first so "Arjun Mehta" wins over "Arjun").
        for original in sorted(self._reverse, key=len, reverse=True):
            text = re.sub(re.escape(original), self._reverse[original], text, flags=re.I)
        text = NAME_LABEL.sub(lambda m: m.group(0) if _PLACEHOLDER.fullmatch(m.group(2).strip())
                              else m.group(1) + self._token("PERSON", m.group(2)), text)
        for kind, rx in PATTERNS:
            text = rx.sub(lambda m, k=kind: self._token(k, m.group(0)), text)
        return text

    def unmask(self, text: str) -> str:
        for tok, original in self.mapping.items():
            text = text.replace(tok, original)
        return text


def scrub_for_logs(text: str) -> str:
    """Last line of defence for log lines: remove anything that looks like contact data."""
    for kind, rx in PATTERNS:
        text = rx.sub(f"[{kind}]", text)
    return text


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip(" .,;:\"'")


def is_grounded(quote: str, sources: list[str], min_len: int = 8) -> bool:
    """True if the quote (whitespace/case-insensitive) occurs in one of the source texts."""
    q = _norm(quote)
    if len(q) < min_len:
        return False
    return any(q in _norm(s) for s in sources)

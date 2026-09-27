"""Pseudonymize people and redact contact data/IDs before conversations leave the machine.

Covers text only. Media files (voice notes, photos) are not redacted: `media.describe_media`
sends them raw to the model, so only run it on data you are allowed to share with that provider.
"""

import re

from genaianalysis.schema import Conversation

URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<![\w$])\+?\d[\d \-().\u202f\u00a0]{5,}\d(?![\w])")
ID = re.compile(r"\b[A-Z]{0,3}\d{6,}[A-Z]?\b")  # cédulas, DNI, account/order numbers
NOT_PHONE = re.compile(  # dates, times and amounts like 1.500.000 are digit runs too
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}(?::\d{2})*)?|\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?"
)


def _phone(m: re.Match[str]) -> str:
    s = m.group(0)
    if NOT_PHONE.fullmatch(s.strip()) or sum(ch.isdigit() for ch in s) < 7 or s.isdigit():
        return s  # bare digit runs are left to ID (a 10-digit run may be a phone or a cédula)
    return "[PHONE]"


def redact(text: str) -> str:
    # ponytail: regex only; misses free-form addresses and names not in the sender list.
    # Swap in an NER-based redactor (e.g. Presidio, or an LLM task) when that matters.
    text = EMAIL.sub("[EMAIL]", URL.sub("[URL]", text))
    return ID.sub("[ID]", PHONE.sub(_phone, text))


def anonymize(
    convs: list[Conversation], roles: dict[str, str] | None = None
) -> tuple[list[Conversation], dict[str, str]]:
    """Replace sender names with `roles[name]` or stable pseudonyms (P1, P2, ...) across all
    conversations, replace mentions of those names (full or first name) in the text, and redact
    contact data. Returns new conversations and the name -> pseudonym mapping (keep it local)."""
    mapping = dict(roles or {})
    for c in convs:
        for m in c.messages:
            mapping.setdefault(m.sender, f"P{len(mapping) - len(roles or {}) + 1}")

    aliases: dict[str, str] = {}
    for name, alias in mapping.items():
        aliases[name] = alias
        first = name.split()[0] if name.split() else ""
        if len(first) >= 3 and not first.startswith("+"):
            aliases.setdefault(first, alias)
    names = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(aliases, key=len, reverse=True)) + r")\b",
        re.I,
    )
    lookup = {k.lower(): v for k, v in aliases.items()}

    def rename(text: str) -> str:
        return names.sub(lambda mt: lookup[mt.group(0).lower()], text) if aliases else text

    def clean(text: str) -> str:
        return redact(rename(text))

    out = [
        c.model_copy(
            update={
                "id": rename(c.id),  # chat ids often embed a contact name
                "meta": {k: rename(v) for k, v in c.meta.items()},
                "messages": [
                    m.model_copy(update={"sender": mapping[m.sender], "text": clean(m.text)})
                    for m in c.messages
                ],
            }
        )
        for c in convs
    ]
    return out, mapping

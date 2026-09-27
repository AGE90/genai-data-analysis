"""WhatsApp "Export chat" (.txt or .zip with media) -> Conversations, one per chat session.

Handles Android (`31/12/23, 9:15 p. m. - Ana: hola`) and iOS (`[31/12/23, 21:15:07] Ana: hola`)
layouts, 12/24h clocks in English and Spanish locales, multi-line messages, omitted media and
attached files. Messages keep the real sender name, so run privacy.anonymize before sending
anything to an API.
"""

import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from genaianalysis.schema import Conversation, Message

LINE = re.compile(
    r"^\u200e?\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)(?:[\s\u202f\u00a0]*(?P<ampm>[aApP])\.?\s?[mM]\.?)?"
    r"\]?\s*(?:-\s*)?(?P<rest>.*)$"
)
# "IMG-20231231-WA0001.jpg (archivo adjunto)" (Android) / "<adjunto: 0001-PHOTO-….jpg>" (iOS)
ATTACHED = re.compile(
    r"^\u200e?(?:(?P<a>\S+\.\w{2,5}) \((?:file attached|archivo adjunto)\)"
    r"|<(?:attached|adjunto): (?P<i>[^>]+)>)(?:\n(?P<caption>.*))?$",
    re.S,
)
OMITTED = re.compile(
    r"^\u200e?<?[\w ]*(?:omitted|omitid[oa])>?$", re.I
)  # "<Media omitted>", "image omitted"


def _dayfirst(dates: list[str]) -> bool:
    """Infer day/month order from the export itself; default day-first (es/most locales)."""
    parts = [re.split(r"[./-]", d) for d in dates]
    if any(len(p[0]) == 4 for p in parts):  # yyyy-mm-dd
        return False
    if any(int(p[1]) > 12 for p in parts):
        return False
    return True


def _parse_ts(date: str, time: str, ampm: str | None, dayfirst: bool) -> datetime:
    a, b, c = (int(x) for x in re.split(r"[./-]", date))
    year, month, day = (a, b, c) if a > 31 else ((c, b, a) if dayfirst else (c, a, b))
    year += 2000 if year < 100 else 0
    h, m, *s = (int(x) for x in time.split(":"))
    if ampm:
        h = h % 12 + (12 if ampm.lower() == "p" else 0)
    return datetime(year, month, day, h, m, s[0] if s else 0)


def parse_whatsapp(
    text: str,
    *,
    chat_id: str = "chat",
    media_dir: Path | None = None,
    session_gap: timedelta | None = timedelta(hours=6),
    dayfirst: bool | None = None,
) -> list[Conversation]:
    """Parse an export's text. A new Conversation starts after `session_gap` of silence."""
    rows: list[dict[str, str | None]] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = LINE.match(line)
        if m:
            rows.append(m.groupdict())
        elif rows:  # continuation of a multi-line message
            rows[-1]["rest"] = f"{rows[-1]['rest']}\n{line}"
    if not rows:
        return []
    df = _dayfirst([str(r["date"]) for r in rows]) if dayfirst is None else dayfirst

    messages: list[Message] = []
    for r in rows:
        sender, sep, body = str(r["rest"]).partition(": ")
        if not sep:  # system line: encryption notice, "X added Y", ...
            # ponytail: system lines containing ": " read as messages; filter by sender if needed
            continue
        media = None
        if a := ATTACHED.match(body):
            name = a["a"] or a["i"]
            media = str(media_dir / name) if media_dir else name
            body = a["caption"] or ""
        elif OMITTED.match(body):
            body = "[media omitted]"
        ts = _parse_ts(str(r["date"]), str(r["time"]), r["ampm"], df)
        messages.append(
            Message(sender=sender.strip("\u200e ~\u202f"), text=body, timestamp=ts, media=media)
        )

    sessions: list[list[Message]] = [[messages[0]]] if messages else []
    for prev, msg in zip(messages, messages[1:], strict=False):
        gap = msg.timestamp - prev.timestamp if msg.timestamp and prev.timestamp else timedelta(0)
        if session_gap is not None and gap > session_gap:
            sessions.append([])
        sessions[-1].append(msg)
    return [
        Conversation(
            id=f"{chat_id}-{i}",
            messages=s,
            meta={"source": "whatsapp", "chat": chat_id, "start": str(s[0].timestamp)},
        )
        for i, s in enumerate(sessions)
    ]


def load_whatsapp(path: Path | str, **kwargs: Any) -> list[Conversation]:
    """Load a `.txt` export, or a `.zip` export (extracted next to it so media paths resolve)."""
    path = Path(path)
    kwargs.setdefault("chat_id", path.stem)  # iOS zips name the text file "_chat.txt"
    if path.suffix == ".zip":
        out = path.with_suffix("")
        with zipfile.ZipFile(path) as z:
            z.extractall(out)  # zipfile strips absolute paths and '..' members
        path = next(p for p in out.rglob("*.txt"))
        kwargs.setdefault("media_dir", path.parent)
    return parse_whatsapp(path.read_text(encoding="utf-8"), **kwargs)

"""Canonical data model: every source is normalized into Conversations, every task is a Task."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Message(BaseModel):
    sender: str  # role ("customer", "agent", "user") or pseudonymized name
    text: str
    timestamp: datetime | None = None
    media: str | None = None  # path to an attached file (image, voice note, video, document)


class Conversation(BaseModel):
    id: str
    messages: list[Message]
    labels: dict[str, str] = {}  # gold labels keyed by task output field, used by eval
    meta: dict[str, str] = {}  # source, language, domain, ...

    def to_text(self) -> str:
        def line(m: Message) -> str:
            ts = f"{m.timestamp.isoformat(' ', 'minutes')} - " if m.timestamp else ""
            text = m.text
            if m.media and not text.startswith("["):  # not yet turned into text by describe_media
                text = f"[attachment: {Path(m.media).name}] {text}".strip()
            return f"{ts}{m.sender}: {text}"

        return "\n".join(line(m) for m in self.messages)


@dataclass(frozen=True)
class Task:
    """What to extract: a pydantic output schema plus instructions.

    Field descriptions double as the questions for non-LLM backends (Jev), so keep them
    self-contained.
    """

    name: str
    output_type: type[BaseModel]
    instructions: str


def save_jsonl(convs: Iterable[Conversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in convs), encoding="utf-8")


def load_jsonl(path: Path) -> list[Conversation]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Conversation.model_validate_json(line) for line in lines if line]

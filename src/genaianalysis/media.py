"""Turn attached media into text once (transcribe / describe / summarize), so every downstream
task stays text-only, cacheable and usable by text-only backends like Jev.

Media is sent raw to the model: see the note in privacy.py before running on private data.
"""

import asyncio
import hashlib
import json
import mimetypes
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from genaianalysis import run
from genaianalysis.schema import Conversation

MIME_OVERRIDES = {".opus": "audio/ogg", ".webp": "image/webp", ".m4a": "audio/mp4"}

# Ported from the legacy Vertex pipeline; output language is now a parameter.
PROMPTS = {
    "audio": "You are an expert at audio transcription. "
    "Transcribe the following audio in {language}.",
    "image": "You are an expert at describing images with certainty and detail. "
    "Describe the following image in {language}.",
    "video": "You are an expert at describing videos with certainty and detail. "
    "Write a short summary in {language} of the following video including the most important "
    "ideas. If there are questions or requests in the video, include them in the summary.",
    "document": "You are an expert at creating document summaries. Write a brief summary in "
    "{language} of the following document accurately describing its content.",
}


def media_kind(path: Path) -> tuple[str, str] | None:
    """(kind, mime) for a supported file, else None."""
    mime = MIME_OVERRIDES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if not mime:
        return None
    major = mime.split("/")[0]
    kind = major if major in ("audio", "image", "video") else "document"
    return kind, mime


async def describe_media(
    convs: Sequence[Conversation],
    model: str | Model,
    *,
    language: str = "Spanish",
    rpm: float | None = None,
    concurrency: int = 8,
) -> list[Conversation]:
    """Return copies where each message with an existing media file gets text
    `[<kind>] <description>` (plus the original caption). Results are cached per file content."""
    agent = Agent(model)
    sem = asyncio.Semaphore(concurrency)
    delay = 60 / rpm if rpm else 0.0
    jobs = [
        (ci, mi, Path(m.media))
        for ci, c in enumerate(convs)
        for mi, m in enumerate(c.messages)
        if m.media and Path(m.media).is_file() and media_kind(Path(m.media))
    ]

    async def one(k: int, path: Path) -> str:
        kind, mime = media_kind(path) or ("", "")
        data = path.read_bytes()
        name = model if isinstance(model, str) else model.model_name
        key = hashlib.sha256(data + f"{name}|{language}|{PROMPTS[kind]}".encode()).hexdigest()[:24]
        cache = run.CACHE_DIR / "media" / f"{key}.json"
        if cache.exists():
            return str(json.loads(cache.read_text())["text"])
        await asyncio.sleep(k * delay)
        try:
            async with sem:
                r = await run.with_retries(
                    lambda: agent.run(
                        [
                            PROMPTS[kind].format(language=language),
                            BinaryContent(data, media_type=mime),
                        ]
                    )
                )
        except Exception as e:  # noqa: BLE001 - keep the conversation, flag the media
            run.logger.error("media %s failed: %s", path.name, e)
            return f"[{kind}] (could not be processed)"
        text = f"[{kind}] {r.output.strip()}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"file": path.name, "text": text}))
        return text

    texts = await asyncio.gather(*(one(k, p) for k, (_, _, p) in enumerate(jobs)))
    out = [c.model_copy(deep=True) for c in convs]
    for (ci, mi, _), text in zip(jobs, texts, strict=True):
        msg = out[ci].messages[mi]
        msg.text = f"{text}\n{msg.text}".strip() if msg.text else text
    return out

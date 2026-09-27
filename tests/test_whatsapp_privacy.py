import asyncio
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

import genaianalysis.run as run_mod
from genaianalysis.ingest.whatsapp import load_whatsapp
from genaianalysis.media import describe_media
from genaianalysis.privacy import anonymize, redact

FIX = Path(__file__).parent / "fixtures"


def test_android_es_sessions_multiline_media():
    convs = load_whatsapp(FIX / "whatsapp_android_es.txt")
    assert [len(c.messages) for c in convs] == [
        5,
        2,
    ]  # 12h+ gap splits sessions; system line dropped
    first = convs[0].messages
    assert first[0].sender == "Ana Gómez" and first[0].timestamp == datetime(2023, 12, 31, 21, 15)
    assert first[0].text.endswith("\nllevo 2 horas esperando")
    assert first[3].media == "IMG-20231231-WA0001.jpg" and first[3].text == "comprobante"
    assert first[4].text == "[media omitted]"
    assert convs[1].messages[0].timestamp == datetime(2024, 1, 1, 10, 2)


def test_ios_en_month_first_and_attachments():
    (c,) = load_whatsapp(FIX / "whatsapp_ios_en.txt")
    assert c.messages[0].timestamp == datetime(2023, 12, 31, 21, 15, 7)
    assert c.messages[0].media.endswith(".opus") and c.messages[0].text == ""
    assert c.messages[1].sender == "Carol" and c.messages[1].text == "sure: see you at 10"
    assert c.messages[2].text == "[media omitted]"


def test_zip_resolves_media_paths(tmp_path):
    z = tmp_path / "WhatsApp Chat - Ana.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(FIX / "whatsapp_android_es.txt", "_chat.txt")
        f.writestr("IMG-20231231-WA0001.jpg", b"\xff\xd8fake")
    convs = load_whatsapp(z)
    assert convs[0].id == "WhatsApp Chat - Ana-0"
    assert Path(convs[0].messages[3].media).is_file()


def test_redact_keeps_dates_and_amounts():
    text = (
        "tel +57 300 123 4567, cc 1032488921, a@b.co, https://x.co/1 el 15-01-2024 pago 1.500.000"
    )
    assert redact(text) == "tel [PHONE], cc [ID], [EMAIL], [URL] el 15-01-2024 pago 1.500.000"


def test_anonymize_senders_mentions_and_ids():
    convs = load_whatsapp(FIX / "whatsapp_android_es.txt", chat_id="Chat con Ana Gómez")
    out, mapping = anonymize(convs, roles={"Soporte Tienda": "agent"})
    assert mapping == {"Soporte Tienda": "agent", "Ana Gómez": "P1"}
    text = "\n".join(c.to_text() for c in out) + " ".join(c.id for c in out)
    assert "Ana" not in text and "1032488921" not in text and "ana.gomez" not in text
    assert "Hola P1" in text and out[0].id == "Chat con P1-0"
    assert convs[0].messages[0].sender == "Ana Gómez"  # originals untouched


def test_describe_media_replaces_text_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "CACHE_DIR", tmp_path / "cache")
    img = tmp_path / "IMG-20231231-WA0001.jpg"
    img.write_bytes(b"\xff\xd8fake")
    convs = load_whatsapp(FIX / "whatsapp_android_es.txt", media_dir=tmp_path)
    out = asyncio.run(describe_media(convs, TestModel(custom_output_text="un recibo")))
    assert out[0].messages[3].text == "[image] un recibo\ncomprobante"
    assert convs[0].messages[3].text == "comprobante"
    again = asyncio.run(describe_media(convs, TestModel(custom_output_text="otro")))  # cache hit
    assert again[0].messages[3].text.startswith("[image] un recibo")


@pytest.mark.parametrize("name", ["whatsapp_android_es.txt", "whatsapp_ios_en.txt"])
def test_to_text_renders_every_message(name):
    for c in load_whatsapp(FIX / name):
        assert len(c.to_text().splitlines()) >= len(c.messages)

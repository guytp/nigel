"""Audio input: record from the USB mic, optionally gain-boost, transcribe via Whisper.

Shelling out to `arecord` and `sox` keeps this independent of sounddevice/
portaudio in the MCP server's venv and avoids contention if the voice_agent
process also wants the mic later.

On a machine without a real mic (laptop), record_wav() raises; listen() will
short-circuit with a synthetic transcript if PICRAWLER_AUDIO_MOCK=1 is set.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DEVICE = os.environ.get("PICRAWLER_AUDIO_DEVICE", "plughw:4,0")
DEFAULT_GAIN_DB = float(os.environ.get("PICRAWLER_AUDIO_GAIN_DB", "20"))
WHISPER_MODEL = os.environ.get("PICRAWLER_WHISPER_MODEL", "whisper-1")


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def record_wav(seconds: float, device: str | None = None, gain_db: float | None = None) -> bytes:
    """Record N seconds from the mic and return gain-boosted WAV bytes (s16le/16kHz/mono)."""
    if os.environ.get("PICRAWLER_AUDIO_MOCK") == "1":
        return _synthetic_silence(seconds)
    if not _tool_available("arecord"):
        raise RuntimeError("arecord not on PATH; cannot capture audio")
    device = device or DEFAULT_DEVICE
    gain_db = DEFAULT_GAIN_DB if gain_db is None else gain_db

    raw = Path(tempfile.mkstemp(suffix=".wav")[1])
    boosted = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        # arecord's -d takes an integer; passing a float string errors.
        duration_int = max(1, int(round(seconds)))
        try:
            subprocess.run(
                [
                    "arecord", "-q", "-D", device,
                    "-f", "S16_LE", "-r", "16000", "-c", "1",
                    "-d", str(duration_int), str(raw),
                ],
                check=True,
                capture_output=True,
                timeout=seconds + 5,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(f"arecord failed (rc={e.returncode}): {stderr}") from e

        if gain_db == 0 or not _tool_available("sox"):
            return raw.read_bytes()
        try:
            subprocess.run(
                ["sox", str(raw), str(boosted), "gain", str(gain_db)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(f"sox failed (rc={e.returncode}): {stderr}") from e
        return boosted.read_bytes()
    finally:
        raw.unlink(missing_ok=True)
        boosted.unlink(missing_ok=True)


def _synthetic_silence(seconds: float) -> bytes:
    """Make a silent WAV of the requested length for tests."""
    import struct
    import wave

    rate = 16000
    nframes = int(seconds * rate)
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(buf.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * nframes)
        return Path(buf.name).read_bytes()
    finally:
        Path(buf.name).unlink(missing_ok=True)


def transcribe(wav_bytes: bytes, api_key: str | None = None) -> str:
    """Send WAV to Whisper and return the transcription text."""
    if os.environ.get("PICRAWLER_AUDIO_MOCK") == "1":
        return os.environ.get("PICRAWLER_AUDIO_MOCK_TRANSCRIPT", "[mock] user said hello")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no OPENAI_API_KEY — set it in /etc/picrawler-mcp.env to enable transcription"
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp.close()
        with open(tmp.name, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=audio_file
            )
        return (result.text or "").strip()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def listen(seconds: float = 5.0) -> dict:
    """Record `seconds` from the mic, transcribe, return text + metadata."""
    t0 = time.monotonic()
    wav = record_wav(seconds)
    rec_ms = (time.monotonic() - t0) * 1000
    t1 = time.monotonic()
    text = transcribe(wav)
    whisper_ms = (time.monotonic() - t1) * 1000
    return {
        "text": text,
        "seconds": seconds,
        "record_ms": round(rec_ms, 1),
        "whisper_ms": round(whisper_ms, 1),
        "bytes": len(wav),
    }


def listen_for_wake_word(
    wake: str = "hey nigel",
    timeout: float = 60.0,
    chunk_seconds: float = 3.0,
    capture_after: float = 4.0,
) -> dict:
    """Stream-ish listen: record chunks, Whisper each, return on wake-word match.

    When the wake word is heard mid-chunk, we capture `capture_after` more
    seconds so we get whatever the user said after the wake word too.
    """
    pattern = re.compile(rf"\b{re.escape(wake.lower())}\b", re.IGNORECASE)
    deadline = time.monotonic() + timeout
    heard: list[str] = []

    while time.monotonic() < deadline:
        wav = record_wav(chunk_seconds)
        text = transcribe(wav)
        heard.append(text)
        if pattern.search(text.lower()):
            followup_wav = record_wav(capture_after)
            followup = transcribe(followup_wav)
            return {
                "woke": True,
                "wake_chunk": text,
                "followup": followup,
                "heard_chunks": heard,
                "timed_out": False,
            }

    return {
        "woke": False,
        "wake_chunk": None,
        "followup": None,
        "heard_chunks": heard,
        "timed_out": True,
    }

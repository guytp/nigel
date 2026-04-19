"""Mock-mode tests for the audio input pipeline.

PICRAWLER_AUDIO_MOCK=1 short-circuits both record_wav (returns synthetic
silence WAV) and transcribe (returns a canned string). Lets us exercise
`listen` and `listen_for_wake_word` control flow without a mic or API key.
"""

from __future__ import annotations

import pytest

from mcp_picrawler import audio_input


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("PICRAWLER_AUDIO_MOCK", "1")
    monkeypatch.setenv("PICRAWLER_AUDIO_MOCK_TRANSCRIPT", "hello nigel how are you")


def test_record_wav_mock_returns_silent_wav():
    data = audio_input.record_wav(seconds=2.0)
    # RIFF header + 16000*2.0 frames of s16le = ~64KB
    assert data.startswith(b"RIFF")
    assert 50_000 < len(data) < 80_000


def test_transcribe_mock_returns_stubbed_text():
    assert audio_input.transcribe(b"irrelevant") == "hello nigel how are you"


def test_listen_structure():
    result = audio_input.listen(seconds=1.0)
    assert set(result) >= {"text", "seconds", "record_ms", "whisper_ms", "bytes"}
    assert "nigel" in result["text"].lower()


def test_wake_word_detected_on_first_chunk():
    result = audio_input.listen_for_wake_word(wake="nigel", timeout=10, chunk_seconds=2)
    assert result["woke"] is True
    assert result["timed_out"] is False
    assert "nigel" in result["wake_chunk"].lower()
    # followup is also captured (will also contain "nigel" in mock)
    assert result["followup"]


def test_wake_word_times_out_when_never_heard(monkeypatch):
    monkeypatch.setenv("PICRAWLER_AUDIO_MOCK_TRANSCRIPT", "just some random talk")
    result = audio_input.listen_for_wake_word(
        wake="nigel", timeout=3, chunk_seconds=1
    )
    assert result["woke"] is False
    assert result["timed_out"] is True
    assert len(result["heard_chunks"]) >= 1


def test_wake_word_matches_case_insensitive(monkeypatch):
    monkeypatch.setenv("PICRAWLER_AUDIO_MOCK_TRANSCRIPT", "Hey NIGEL, come here")
    result = audio_input.listen_for_wake_word(
        wake="nigel", timeout=5, chunk_seconds=1
    )
    assert result["woke"] is True

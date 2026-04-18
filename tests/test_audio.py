"""Audio I/O smoke tests — requires portaudio + a real audio device.

Verifies AudioIO can open input + output streams, write and drain output,
and flush on barge-in. Skipped when no default device is available (CI).
"""

from __future__ import annotations

import time

import pytest

from voice_agent.audio import AudioIO, BLOCK_SAMPLES, SAMPLE_RATE


def _has_devices() -> bool:
    try:
        import sounddevice as sd

        sd.query_devices(kind="input")
        sd.query_devices(kind="output")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_devices(), reason="no default audio device (CI / no portaudio)"
)


def test_audio_can_start_and_stop():
    audio = AudioIO()
    audio.start()
    try:
        time.sleep(0.1)  # give streams a moment
    finally:
        audio.stop()


def test_audio_output_buffer_drains():
    audio = AudioIO()
    audio.start()
    try:
        silence = b"\x00\x00" * BLOCK_SAMPLES * 3
        audio.enqueue_output(silence)
        # outbuf has 3 blocks; callback drains ~1 block per BLOCK_MS
        time.sleep(0.2)
        # internal buffer should be empty or near-empty after 200ms
        assert len(audio._out_buf) < len(silence)
    finally:
        audio.stop()


def test_flush_output_clears_buffer():
    audio = AudioIO()
    audio.start()
    try:
        noise = b"\x7f\x00" * BLOCK_SAMPLES * 5
        audio.enqueue_output(noise)
        assert len(audio._out_buf) > 0
        audio.flush_output()
        assert len(audio._out_buf) == 0
    finally:
        audio.stop()


def test_audio_module_constants_match_openai_realtime():
    """OpenAI Realtime's pcm16 is 24kHz mono — our defaults must match."""
    assert SAMPLE_RATE == 24000
    assert BLOCK_SAMPLES == 960  # 40ms at 24kHz

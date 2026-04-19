"""Mic + speaker I/O with rate conversion between hardware and OpenAI Realtime.

OpenAI Realtime's `pcm16` codec is fixed at 24kHz mono. Most USB audio
adapters don't natively support 24kHz — a TI PCM2902 only does 44.1/48kHz.
So we run the hardware streams at whatever the device supports (default
48kHz) and resample at the boundary. 48→24 and 24→48 are exact 2:1 ratios
so a cheap average / duplicate suffices for speech.

Env overrides (the service reads these before AudioIO starts):
  VOICE_INPUT_DEVICE   — sounddevice index or name for the mic (default: system)
  VOICE_OUTPUT_DEVICE  — sounddevice index or name for the speaker (default: system)
  VOICE_HW_INPUT_RATE  — mic hardware sample rate in Hz (default 48000)
  VOICE_HW_OUTPUT_RATE — speaker hardware sample rate in Hz (default 48000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading

log = logging.getLogger(__name__)

OPENAI_RATE = 24000  # pcm16 is fixed at 24kHz mono in OpenAI Realtime
BLOCK_MS = 40

HW_INPUT_RATE = int(os.environ.get("VOICE_HW_INPUT_RATE", "48000"))
HW_OUTPUT_RATE = int(os.environ.get("VOICE_HW_OUTPUT_RATE", "48000"))

_DEFAULT_INPUT_DEVICE: str | int | None = os.environ.get("VOICE_INPUT_DEVICE")
_DEFAULT_OUTPUT_DEVICE: str | int | None = os.environ.get("VOICE_OUTPUT_DEVICE")


def _coerce_device(val):
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return val  # fall back to device-name string


def _resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Cheap PCM16 mono resampler. Exact halving/doubling when src/dst differ by 2x."""
    if src_rate == dst_rate or not pcm:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if src_rate == 2 * dst_rate:
        # decimate: average consecutive pairs
        if len(samples) % 2 == 1:
            samples = samples[:-1]
        out = ((samples[0::2].astype(np.int32) + samples[1::2].astype(np.int32)) // 2)
        return out.astype(np.int16).tobytes()
    if dst_rate == 2 * src_rate:
        # duplicate each sample
        return np.repeat(samples, 2).astype(np.int16).tobytes()
    # general-case linear interp (acceptable for speech; rarely hit)
    new_len = int(len(samples) * dst_rate / src_rate)
    idx = np.linspace(0, len(samples) - 1, new_len)
    out = np.interp(idx, np.arange(len(samples)), samples).astype(np.int16)
    return out.tobytes()


class AudioIO:
    """Callback-driven sounddevice streams with a threadsafe output buffer.

    Public API speaks in OpenAI-rate bytes (24kHz pcm16). Internally the hw
    streams run at device-native rates; resampling happens at the boundary.
    """

    def __init__(
        self,
        input_device=None,
        output_device=None,
        hw_input_rate: int | None = None,
        hw_output_rate: int | None = None,
    ) -> None:
        self._input_device = _coerce_device(input_device if input_device is not None else _DEFAULT_INPUT_DEVICE)
        self._output_device = _coerce_device(output_device if output_device is not None else _DEFAULT_OUTPUT_DEVICE)
        self._hw_in_rate = hw_input_rate or HW_INPUT_RATE
        self._hw_out_rate = hw_output_rate or HW_OUTPUT_RATE
        self._in_q: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()
        self._in_stream = None
        self._out_stream = None

    def start(self) -> None:
        import sounddevice as sd

        def in_cb(indata, frames, time_info, status):
            if status:
                log.debug("input status: %s", status)
            try:
                self._in_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        def out_cb(outdata, frames, time_info, status):
            if status:
                log.debug("output status: %s", status)
            needed = frames * 2  # s16 mono
            with self._out_lock:
                if len(self._out_buf) >= needed:
                    outdata[:] = bytes(self._out_buf[:needed])
                    del self._out_buf[:needed]
                else:
                    have = len(self._out_buf)
                    outdata[:have] = bytes(self._out_buf)
                    outdata[have:] = b"\x00" * (needed - have)
                    self._out_buf.clear()

        in_blocksize = self._hw_in_rate * BLOCK_MS // 1000
        out_blocksize = self._hw_out_rate * BLOCK_MS // 1000

        self._in_stream = sd.RawInputStream(
            samplerate=self._hw_in_rate,
            channels=1,
            dtype="int16",
            blocksize=in_blocksize,
            device=self._input_device,
            callback=in_cb,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=self._hw_out_rate,
            channels=1,
            dtype="int16",
            blocksize=out_blocksize,
            device=self._output_device,
            callback=out_cb,
        )
        self._in_stream.start()
        self._out_stream.start()
        log.info(
            "audio streams started — in=%dHz@dev=%r  out=%dHz@dev=%r  (openai=24kHz)",
            self._hw_in_rate, self._input_device,
            self._hw_out_rate, self._output_device,
        )

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception as e:
                    log.debug("audio close: %s", e)
        self._in_stream = None
        self._out_stream = None

    async def read_chunk(self) -> bytes:
        """Pop one block of mic PCM, resampled to OpenAI's 24kHz."""
        hw_pcm = await asyncio.to_thread(self._in_q.get)
        return _resample_int16(hw_pcm, self._hw_in_rate, OPENAI_RATE)

    def enqueue_output(self, pcm24k_bytes: bytes) -> None:
        """Accept 24kHz PCM from OpenAI; resample to speaker hw rate and queue."""
        hw_pcm = _resample_int16(pcm24k_bytes, OPENAI_RATE, self._hw_out_rate)
        with self._out_lock:
            self._out_buf.extend(hw_pcm)

    def flush_output(self) -> None:
        with self._out_lock:
            self._out_buf.clear()

    def flush_input(self) -> None:
        """Drop any queued mic chunks. Call this when exiting a mute window
        so echo tail captured during bot speech doesn't get sent upstream."""
        try:
            while True:
                self._in_q.get_nowait()
        except queue.Empty:
            pass


# Legacy names kept for tests that imported them.
SAMPLE_RATE = OPENAI_RATE
BLOCK_SAMPLES = OPENAI_RATE * BLOCK_MS // 1000

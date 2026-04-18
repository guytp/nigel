"""Mic + speaker I/O for the Realtime session.

OpenAI Realtime uses `pcm16`: 16-bit signed PCM, 24kHz, mono. We run input and
output at that rate so there's no resampling stage in the hot path.

Design: callback-driven streams with threadsafe queues. The asyncio main loop
pulls mic chunks via to_thread; the output callback drains a queue into the
device buffer. `flush_output()` is used for barge-in — when the user starts
speaking we drop any pending TTS audio immediately.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
BLOCK_MS = 40  # 40ms blocks → 960 samples at 24kHz
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000


class AudioIO:
    def __init__(self, input_device: int | str | None = None, output_device: int | str | None = None) -> None:
        self._input_device = input_device
        self._output_device = output_device
        self._in_q: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()
        self._in_stream = None
        self._out_stream = None

    def start(self) -> None:
        import sounddevice as sd  # local import — keeps main module importable without portaudio

        def in_cb(indata, frames, time_info, status):
            if status:
                log.debug("input status: %s", status)
            try:
                self._in_q.put_nowait(bytes(indata))
            except queue.Full:
                pass  # drop oldest-equivalent; OpenAI tolerates small gaps

        def out_cb(outdata, frames, time_info, status):
            if status:
                log.debug("output status: %s", status)
            needed = frames * 2  # 16-bit samples, mono
            with self._out_lock:
                if len(self._out_buf) >= needed:
                    outdata[:] = bytes(self._out_buf[:needed])
                    del self._out_buf[:needed]
                else:
                    have = len(self._out_buf)
                    outdata[:have] = bytes(self._out_buf)
                    outdata[have:] = b"\x00" * (needed - have)
                    self._out_buf.clear()

        self._in_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SAMPLES,
            device=self._input_device,
            callback=in_cb,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SAMPLES,
            device=self._output_device,
            callback=out_cb,
        )
        self._in_stream.start()
        self._out_stream.start()
        log.info("audio streams started @ %dHz mono pcm16", SAMPLE_RATE)

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
        """Blocking-in-thread pop of the next mic PCM block."""
        return await asyncio.to_thread(self._in_q.get)

    def enqueue_output(self, pcm16_bytes: bytes) -> None:
        with self._out_lock:
            self._out_buf.extend(pcm16_bytes)

    def flush_output(self) -> None:
        """Barge-in: discard any queued TTS so the bot shuts up immediately."""
        with self._out_lock:
            self._out_buf.clear()

"""Tiered local-vision stack.

Three tiers, each progressively more expensive. The tools layer decides which
tier to invoke — callers are expected to prefer the cheapest tier that answers
their question, and only reach for a full image (T4) when the cheaper signals
warrant it.

    T0  motion delta + perceptual hash  ~5ms    always-on,    pure OpenCV
    T2  YOLOv8n object detection        ~200ms  optional,     ultralytics extra
    T3  Moondream2 caption              ~1-2s   optional,     caption extra

T0 is free once OpenCV is in. T2 and T3 do lazy model load on first call; if
the required package isn't installed, the method raises a clear error that the
MCP tool surfaces as a normal tool error.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class ScanResult:
    motion: float  # 0.0 = identical to previous frame, ~1.0 = totally different
    phash: str
    objects: list[Detection] = field(default_factory=list)
    elapsed_ms: float = 0.0
    tiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion": round(self.motion, 4),
            "phash": self.phash,
            "objects": [
                {"label": d.label, "conf": round(d.conf, 3), "bbox": list(d.bbox)}
                for d in self.objects
            ],
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tiers": self.tiers,
        }


# Bare filename so ultralytics auto-downloads and caches it — passing a
# prefixed path (e.g. "models/yolov8n.pt") doesn't survive service restart:
# ultralytics writes the download to CWD as `yolov8n.pt`, then on restart
# can't find it at the prefixed path and re-downloads.
DEFAULT_YOLO_PATH = "yolov8n.pt"


class VisionStack:
    def __init__(
        self,
        yolo_model: str = DEFAULT_YOLO_PATH,
        caption_model: str = "vikhyatk/moondream2",
    ) -> None:
        self._yolo_model_name = yolo_model
        self._caption_model_name = caption_model
        self._prev_gray = None  # type: ignore[assignment]
        self._yolo = None
        self._moondream = None
        self._moondream_tokenizer = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ T0

    def _motion(self, frame_bgr) -> float:
        """Mean absolute per-pixel delta vs previous frame, normalised 0..1."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 48))
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        delta = np.mean(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))) / 255.0
        self._prev_gray = gray
        return float(delta)

    def _phash(self, frame_bgr) -> str:
        """64-bit perceptual hash — cheap scene fingerprint."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)
        low = dct[:8, :8].flatten()[1:]  # drop DC term
        med = np.median(low)
        bits = (low > med).astype(np.uint8)
        value = 0
        for b in bits[:64]:
            value = (value << 1) | int(b)
        return f"{value:016x}"

    # ------------------------------------------------------------------ T2

    def _load_yolo(self):
        if self._yolo is not None:
            return self._yolo
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "object detection unavailable — install with `pip install '.[vision]'`"
            ) from e
        with self._lock:
            if self._yolo is None:
                log.info("loading yolo model %s", self._yolo_model_name)
                self._yolo = YOLO(self._yolo_model_name)
        return self._yolo

    def _detect_objects(self, frame_bgr, conf: float = 0.35) -> list[Detection]:
        yolo = self._load_yolo()
        results = yolo.predict(frame_bgr, conf=conf, verbose=False)
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls = int(box.cls[0])
                c = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label=names[cls], conf=c, bbox=(x1, y1, x2, y2)))
        return out

    # ------------------------------------------------------------------ T3

    def _load_moondream(self):
        if self._moondream is not None:
            return self._moondream, self._moondream_tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "captioning unavailable — install with `pip install '.[caption]'`"
            ) from e
        with self._lock:
            if self._moondream is None:
                log.info("loading caption model %s (first call; slow)", self._caption_model_name)
                self._moondream_tokenizer = AutoTokenizer.from_pretrained(
                    self._caption_model_name, trust_remote_code=True
                )
                self._moondream = AutoModelForCausalLM.from_pretrained(
                    self._caption_model_name, trust_remote_code=True
                )
        return self._moondream, self._moondream_tokenizer

    def caption(self, frame_bgr, prompt: str | None = None) -> str:
        from PIL import Image

        model, tokenizer = self._load_moondream()
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        enc = model.encode_image(pil)
        q = prompt or "Describe this scene in one short sentence."
        return model.answer_question(enc, q, tokenizer).strip()

    # ------------------------------------------------------------------ public

    def scan(
        self,
        frame_bgr,
        include_objects: bool = True,
        object_conf: float = 0.35,
    ) -> ScanResult:
        """T0 always, T2 on request. Returns a terse JSON-friendly result."""
        t0 = time.perf_counter()
        tiers = ["motion", "phash"]
        motion = self._motion(frame_bgr)
        phash = self._phash(frame_bgr)
        objects: list[Detection] = []
        if include_objects:
            try:
                objects = self._detect_objects(frame_bgr, conf=object_conf)
                tiers.append("objects")
            except RuntimeError as e:
                log.warning("object detection skipped: %s", e)
        elapsed = (time.perf_counter() - t0) * 1000
        return ScanResult(
            motion=motion,
            phash=phash,
            objects=objects,
            elapsed_ms=elapsed,
            tiers=tiers,
        )

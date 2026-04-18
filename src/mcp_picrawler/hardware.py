"""Hardware abstraction over picrawler + robot_hat + vilib.

Import-time autodetect: if the picrawler libs aren't present we fall back to a
mock. The mock is there so the MCP server runs identically on a laptop.
"""

from __future__ import annotations

import io
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)


BUILTIN_ACTIONS: tuple[str, ...] = (
    "forward",
    "backward",
    "turn left",
    "turn right",
    "sit",
    "stand",
    "wave",
    "twist",
    "beckon",
    "push-up",
    "look up",
    "look down",
)

DETECTIONS: tuple[str, ...] = ("face", "color", "qr", "gesture", "object")


@dataclass
class Detection:
    kind: str
    data: dict


@dataclass
class State:
    last_action: str = "idle"
    vision: dict[str, bool] = field(default_factory=dict)
    target_color: str | None = None


class Hardware(Protocol):
    kind: str
    state: State

    def do_action(self, name: str, steps: int = 1, speed: int = 90) -> None: ...
    def stop(self) -> None: ...
    def read_distance_cm(self) -> float: ...
    def latest_frame_bgr(self):  # -> np.ndarray (HxWx3, uint8, BGR)
        ...
    def snapshot_jpeg(self) -> bytes: ...
    def set_vision(self, feature: str, enabled: bool) -> None: ...
    def read_detections(self) -> list[Detection]: ...
    def set_target_color(self, color: str) -> None: ...
    def speak(self, text: str) -> None: ...
    def stream_url(self) -> str: ...


class MockHardware:
    """Runs on any machine. Logs commands and synthesises sensor output.

    Set PICRAWLER_MOCK_CAMERA=webcam to pull frames from the default webcam
    (useful for Tier-C testing of the vision stack before the Pi arrives).
    Falls back to synthetic frames if the webcam can't open.
    """

    kind = "mock"

    def __init__(self) -> None:
        self.state = State()
        self._webcam = None  # None=untried, False=failed, VideoCapture=open

    def do_action(self, name: str, steps: int = 1, speed: int = 90) -> None:
        if name not in BUILTIN_ACTIONS:
            raise ValueError(f"unknown action: {name}")
        log.info("mock do_action %s steps=%d speed=%d", name, steps, speed)
        self.state.last_action = name
        time.sleep(min(0.05 * steps, 0.5))

    def stop(self) -> None:
        log.info("mock stop")
        self.state.last_action = "stop"

    def read_distance_cm(self) -> float:
        return round(random.uniform(15.0, 120.0), 1)

    def latest_frame_bgr(self):
        if os.environ.get("PICRAWLER_MOCK_CAMERA", "").lower() == "webcam":
            frame = self._webcam_frame_bgr()
            if frame is not None:
                return frame
        return self._synthetic_frame_bgr()

    def _webcam_frame_bgr(self):
        """Pull a frame from the default webcam, downscaled to PiCrawler resolution."""
        if self._webcam is False:
            return None
        import cv2

        if self._webcam is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                log.warning("PICRAWLER_MOCK_CAMERA=webcam but device 0 did not open")
                self._webcam = False
                return None
            self._webcam = cap
        ok, frame = self._webcam.read()
        if not ok or frame is None:
            return None
        return cv2.resize(frame, (320, 240))

    def _synthetic_frame_bgr(self):
        import numpy as np
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (320, 240), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), "MOCK CAMERA", fill=(240, 240, 240))
        draw.text((12, 32), f"last: {self.state.last_action}", fill=(180, 220, 180))
        draw.text((12, 52), f"dist: {self.read_distance_cm()} cm", fill=(180, 220, 180))
        # simulate a little drift so motion-delta has something to chew on
        jitter = random.randint(0, 8)
        draw.rectangle((200 + jitter, 180, 240 + jitter, 220), fill=(200, 60, 60))
        rgb = np.array(img)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR

    def snapshot_jpeg(self) -> bytes:
        from PIL import Image

        bgr = self.latest_frame_bgr()
        rgb = bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def set_vision(self, feature: str, enabled: bool) -> None:
        if feature not in DETECTIONS:
            raise ValueError(f"unknown detection: {feature}")
        self.state.vision[feature] = enabled

    def read_detections(self) -> list[Detection]:
        out: list[Detection] = []
        for feat, on in self.state.vision.items():
            if not on:
                continue
            out.append(Detection(kind=feat, data={"n": random.randint(0, 2)}))
        return out

    def set_target_color(self, color: str) -> None:
        self.state.target_color = color

    def speak(self, text: str) -> None:
        log.info("mock speak: %s", text)

    def stream_url(self) -> str:
        return "mock://no-stream"


class RealHardware:
    """Wraps picrawler + robot_hat + vilib on the Pi."""

    kind = "real"

    def __init__(self, stream_host: str = "0.0.0.0", stream_port: int = 9000) -> None:
        from picrawler import Picrawler  # type: ignore[import-not-found]
        from robot_hat import TTS, Ultrasonic, Pin  # type: ignore[import-not-found]
        from vilib import Vilib  # type: ignore[import-not-found]

        self._crawler = Picrawler()
        self._tts = TTS()
        self._ultrasonic = Ultrasonic(Pin("D2"), Pin("D3"))
        self._vilib = Vilib
        self._stream_host = stream_host
        self._stream_port = stream_port
        self.state = State()

        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)

    def do_action(self, name: str, steps: int = 1, speed: int = 90) -> None:
        if name not in BUILTIN_ACTIONS:
            raise ValueError(f"unknown action: {name}")
        self._crawler.do_action(name, steps, speed)
        self.state.last_action = name

    def stop(self) -> None:
        self._crawler.do_action("stand", 1, 80)
        self.state.last_action = "stop"

    def read_distance_cm(self) -> float:
        return float(self._ultrasonic.read())

    def latest_frame_bgr(self):
        frame = self._vilib.img
        if frame is None:
            raise RuntimeError("camera frame not ready")
        return frame  # vilib already gives us BGR numpy

    def snapshot_jpeg(self) -> bytes:
        import cv2  # type: ignore[import-not-found]

        frame = self.latest_frame_bgr()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def set_vision(self, feature: str, enabled: bool) -> None:
        switch = {
            "face": self._vilib.human_detect_switch,
            "color": self._vilib.color_detect_switch,
            "qr": self._vilib.qrcode_detect_switch,
            "gesture": self._vilib.gesture_detect_switch,
            "object": self._vilib.object_follow_switch,
        }.get(feature)
        if switch is None:
            raise ValueError(f"unknown detection: {feature}")
        switch(enabled)
        self.state.vision[feature] = enabled

    def read_detections(self) -> list[Detection]:
        out: list[Detection] = []
        if self.state.vision.get("face"):
            n = self._vilib.detect_obj_parameter.get("human_n", 0)
            out.append(Detection("face", {"n": n}))
        if self.state.vision.get("color"):
            out.append(
                Detection(
                    "color",
                    {
                        "n": self._vilib.detect_obj_parameter.get("color_n", 0),
                        "x": self._vilib.detect_obj_parameter.get("color_x", 0),
                        "y": self._vilib.detect_obj_parameter.get("color_y", 0),
                    },
                )
            )
        if self.state.vision.get("qr"):
            out.append(
                Detection("qr", {"data": self._vilib.detect_obj_parameter.get("qr_data", "")})
            )
        return out

    def set_target_color(self, color: str) -> None:
        self._vilib.detect_color_name(color)
        self.state.target_color = color

    def speak(self, text: str) -> None:
        self._tts.say(text)

    def stream_url(self) -> str:
        return f"http://{self._stream_host}:{self._stream_port}/mjpg"


def get_hardware() -> Hardware:
    """Try real hardware; fall back to mock if the Pi libs aren't importable."""
    try:
        return RealHardware()
    except ImportError as e:
        log.warning("picrawler libs not available (%s); using mock hardware", e)
        return MockHardware()
    except Exception as e:  # hardware present but init failed
        log.warning("real hardware init failed (%s); using mock", e)
        return MockHardware()

"""Hardware abstraction over picrawler + robot_hat + vilib.

Import-time autodetect: if the picrawler libs aren't present we fall back to a
mock. The mock is there so the MCP server runs identically on a laptop.
"""

from __future__ import annotations

import io
import logging
import os
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)


# Must stay in sync with the set picrawler.picrawler.MoveList exposes. See
# https://github.com/sunfounder/picrawler/blob/main/picrawler/picrawler.py
BUILTIN_ACTIONS: tuple[str, ...] = (
    "forward",
    "backward",
    "turn left",
    "turn right",
    "turn left angle",
    "turn right angle",
    "stand",
    "sit",
    "ready",
    "push up",
    "wave",
    "look left",
    "look right",
    "look up",
    "look down",
    "dance",
)

# vilib detection toggles. Omitted:
#  - "object": vilib's object_detect needs a TFLite model + label set we don't
#    ship. Our Tier-C YOLO stack (vision.VisionStack) covers this already.
#  - "gesture": vilib's hands_detect_switch depends on mediapipe, which has no
#    wheel for cp313 aarch64 (Python 3.13 on Pi OS). Build-from-source is hours.
DETECTIONS: tuple[str, ...] = ("face", "color", "qr", "traffic")


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
        log.info("mock do_action %s step=%d speed=%d", name, steps, speed)
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
    """Wraps picrawler + robot_hat + vilib on the Pi.

    API shapes verified against:
      picrawler  main  — picrawler/picrawler.py
      robot-hat  v2.0  — robot_hat/modules.py (Ultrasonic), tts.py
      vilib      picamera2 — vilib/vilib.py
    """

    kind = "real"

    # Ultrasonic smoothing config
    ULTRASONIC_POLL_HZ = 10
    ULTRASONIC_WINDOW_S = 7

    def __init__(self, stream_host: str = "0.0.0.0", stream_port: int = 9000) -> None:
        from picrawler import Picrawler  # type: ignore[import-not-found]
        from robot_hat import TTS, Ultrasonic, Pin  # type: ignore[import-not-found]
        from vilib import Vilib  # type: ignore[import-not-found]

        self._crawler = Picrawler()
        self._tts = TTS()
        # Ultrasonic requires Pin-wrapped args; raw strings raise TypeError.
        # D2=trig, D3=echo per picrawler/examples/avoid.py.
        self._ultrasonic = Ultrasonic(Pin("D2"), Pin("D3"))
        self._vilib = Vilib
        self._stream_host = stream_host
        self._stream_port = stream_port
        self.state = State()

        # Enable the robot_hat speaker amplifier so TTS is audible.
        try:
            from robot_hat.utils import enable_speaker  # type: ignore[import-not-found]

            enable_speaker()
        except Exception as e:
            log.warning("enable_speaker() failed (%s) — TTS may be silent", e)

        # Background ultrasonic reader — HC-SR04 is flaky (timeouts, -1/-2
        # sentinels on random reads). A dedicated thread polls at a steady
        # rate into a rolling window so read_distance_cm can return a smoothed
        # median that hides transient errors.
        window_size = max(3, self.ULTRASONIC_POLL_HZ * self.ULTRASONIC_WINDOW_S)
        self._distance_samples: deque[float] = deque(maxlen=window_size)
        self._distance_lock = threading.Lock()
        self._distance_stop = threading.Event()
        self._distance_thread = threading.Thread(
            target=self._ultrasonic_poll_loop,
            name="ultrasonic-poller",
            daemon=True,
        )
        self._distance_thread.start()

        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)

    def _ultrasonic_poll_loop(self) -> None:
        """Poll ultrasonic at ULTRASONIC_POLL_HZ; drop sentinels."""
        period = 1.0 / self.ULTRASONIC_POLL_HZ
        while not self._distance_stop.is_set():
            try:
                raw = float(self._ultrasonic.read())
            except Exception as e:  # sensor driver hiccup
                log.debug("ultrasonic raw read failed: %s", e)
                raw = -1.0
            if raw > 0 and raw < 500:  # valid range: 1–500cm
                with self._distance_lock:
                    self._distance_samples.append(raw)
            self._distance_stop.wait(period)

    def do_action(self, name: str, steps: int = 1, speed: int = 90) -> None:
        if name not in BUILTIN_ACTIONS:
            raise ValueError(f"unknown action: {name}")
        # picrawler uses `step=` (singular), not `steps=`.
        self._crawler.do_action(name, step=steps, speed=speed)
        self.state.last_action = name

    def stop(self) -> None:
        self._crawler.do_action("stand", step=1, speed=80)
        self.state.last_action = "stop"

    def read_distance_cm(self) -> float:
        """Return the median of recent valid ultrasonic readings.

        The background thread poller drops sentinels (-1/-2) and out-of-range
        values. If no valid reading has landed in the current window (sensor
        is fully broken), we return -1 so callers can still tell.
        """
        with self._distance_lock:
            if not self._distance_samples:
                return -1.0
            # median is robust to outliers (the one bad read in 50).
            return round(statistics.median(self._distance_samples), 2)

    def latest_frame_bgr(self, retry_budget_s: float = 3.0):
        """Return the latest camera frame as a numpy BGR array.

        Vilib.img is a multiprocessing Manager().list() until the camera loop
        reassigns it to a real ndarray. We poll briefly so the first call
        after service boot works instead of hard-failing.
        """
        import numpy as np

        deadline = time.monotonic() + retry_budget_s
        while True:
            frame = self._vilib.img
            if frame is not None and isinstance(frame, np.ndarray):
                return frame  # picamera2 RGB888 buffer is BGR-ordered
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"camera frame not ready after {retry_budget_s}s — "
                    "check CSI cable, vilib logs, and that camera_start() completed"
                )
            time.sleep(0.1)

    def snapshot_jpeg(self) -> bytes:
        import cv2  # type: ignore[import-not-found]

        frame = self.latest_frame_bgr()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def set_vision(self, feature: str, enabled: bool) -> None:
        if feature not in DETECTIONS:
            raise ValueError(f"unknown detection: {feature}")

        # Color is special: vilib's color_detect(name) both enables it AND
        # sets the target. "close" (or close_color_detection()) disables.
        if feature == "color":
            if enabled:
                target = self.state.target_color or "red"
                self._vilib.color_detect(target)
            else:
                self._vilib.close_color_detection()
            self.state.vision["color"] = enabled
            return

        switch = {
            "face": self._vilib.face_detect_switch,
            "qr": self._vilib.qrcode_detect_switch,
            "traffic": self._vilib.traffic_detect_switch,
        }[feature]
        switch(enabled)
        self.state.vision[feature] = enabled

    def read_detections(self) -> list[Detection]:
        params = self._vilib.detect_obj_parameter
        out: list[Detection] = []
        if self.state.vision.get("face"):
            out.append(Detection("face", {"n": params.get("human_n", 0)}))
        if self.state.vision.get("color"):
            out.append(
                Detection(
                    "color",
                    {
                        "n": params.get("color_n", 0),
                        "x": params.get("color_x", 0),
                        "y": params.get("color_y", 0),
                    },
                )
            )
        if self.state.vision.get("qr"):
            out.append(Detection("qr", {"data": params.get("qr_data", "")}))
        return out

    def set_target_color(self, color: str) -> None:
        # No separate "set colour name" API — color_detect(name) is the setter.
        # Only call it if color detection is currently enabled, otherwise we'd
        # implicitly enable it here.
        self.state.target_color = color
        if self.state.vision.get("color"):
            self._vilib.color_detect(color)

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

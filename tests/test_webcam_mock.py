"""Tier-C: real webcam feed as MockHardware's camera.

Only runs when PICRAWLER_TEST_WEBCAM=1 — normally skipped because CI machines
have no camera and the Mac needs user consent to open one.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mcp_picrawler.hardware import MockHardware


pytestmark = pytest.mark.skipif(
    os.environ.get("PICRAWLER_TEST_WEBCAM") != "1",
    reason="needs a webcam; enable with PICRAWLER_TEST_WEBCAM=1",
)


def test_webcam_frame_looks_like_a_photo(monkeypatch):
    monkeypatch.setenv("PICRAWLER_MOCK_CAMERA", "webcam")
    hw = MockHardware()
    frame = hw.latest_frame_bgr()
    assert frame.shape == (240, 320, 3)
    # synthetic frames are nearly constant; real webcam should have variance
    assert float(np.std(frame)) > 10.0


def test_yolo_on_webcam_frame():
    """Smoke: YOLO runs on a real frame and doesn't crash. Objects optional."""
    os.environ["PICRAWLER_MOCK_CAMERA"] = "webcam"
    from mcp_picrawler.vision import VisionStack

    hw = MockHardware()
    vs = VisionStack()
    # warm: first call loads YOLO
    r = vs.scan(hw.latest_frame_bgr(), include_objects=True)
    assert "objects" in r.tiers
    # if you're in front of a laptop, YOLO will likely find `person`; but we
    # don't assert that — just that detection executes without error

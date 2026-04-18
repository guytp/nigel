"""Unit tests for MockHardware.

Goal: the mock honours the `Hardware` protocol so the MCP server behaves
identically on laptop and Pi except for physics. Every method must return the
type the protocol advertises.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcp_picrawler.hardware import (
    BUILTIN_ACTIONS,
    DETECTIONS,
    MockHardware,
    get_hardware,
)


@pytest.fixture
def hw():
    return MockHardware()


def test_get_hardware_falls_back_to_mock_locally():
    """On a laptop the picrawler libs aren't importable → we get a mock."""
    h = get_hardware()
    assert h.kind == "mock"


def test_kind_is_mock(hw):
    assert hw.kind == "mock"


@pytest.mark.parametrize("action", BUILTIN_ACTIONS)
def test_do_action_accepts_all_builtins(hw, action):
    hw.do_action(action, steps=1, speed=50)
    assert hw.state.last_action == action


def test_do_action_rejects_unknown(hw):
    with pytest.raises(ValueError):
        hw.do_action("backflip")


def test_stop_returns_to_stop_state(hw):
    hw.do_action("forward", steps=1)
    hw.stop()
    assert hw.state.last_action == "stop"


def test_read_distance_in_plausible_range(hw):
    for _ in range(20):
        d = hw.read_distance_cm()
        assert isinstance(d, float)
        assert 0 < d < 500


def test_latest_frame_bgr_shape(hw):
    frame = hw.latest_frame_bgr()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3  # BGR


def test_snapshot_jpeg_is_valid_jpeg(hw):
    blob = hw.snapshot_jpeg()
    assert isinstance(blob, bytes)
    assert blob[:2] == b"\xff\xd8"  # JPEG SOI
    assert blob[-2:] == b"\xff\xd9"  # JPEG EOI
    assert 500 < len(blob) < 200_000


@pytest.mark.parametrize("feat", DETECTIONS)
def test_set_vision_toggles_all_features(hw, feat):
    hw.set_vision(feat, True)
    assert hw.state.vision[feat] is True
    hw.set_vision(feat, False)
    assert hw.state.vision[feat] is False


def test_set_vision_rejects_unknown(hw):
    with pytest.raises(ValueError):
        hw.set_vision("xray", True)


def test_read_detections_reflects_toggles(hw):
    hw.set_vision("face", True)
    hw.set_vision("color", True)
    out = hw.read_detections()
    kinds = {d.kind for d in out}
    assert "face" in kinds
    assert "color" in kinds


def test_set_target_color_persists(hw):
    hw.set_target_color("red")
    assert hw.state.target_color == "red"


def test_speak_does_not_raise(hw):
    hw.speak("hello world")  # no-op on mock


def test_stream_url_is_string(hw):
    assert isinstance(hw.stream_url(), str)

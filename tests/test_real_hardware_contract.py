"""Regression tests pinning the RealHardware → SunFounder API contract.

We can't import the real `picrawler`/`robot_hat`/`vilib` on the dev machine
(they're Pi-only), so we fake those modules in sys.modules with mocks that
record calls, then instantiate RealHardware and verify it invokes them with
the exact shape the real libraries expect.

These tests exist because real APIs have drifted on us before (see bug
report in the 2026-04 verification pass):
 - picrawler uses `step=` not `steps=`
 - robot_hat has NO Ultrasonic at the top level? Actually yes, re-exported
 - vilib renamed toggles (human_detect → face_detect, color_detect_switch
   replaced by color_detect(name), gesture → hands, object_follow → object_detect)
 - vilib has no `detect_color_name` — color_detect(name) is the setter
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_sunfounder(monkeypatch):
    """Inject fake picrawler/robot_hat/vilib modules so RealHardware imports cleanly."""
    fake_picrawler = types.ModuleType("picrawler")
    fake_picrawler.Picrawler = MagicMock(name="Picrawler")
    monkeypatch.setitem(sys.modules, "picrawler", fake_picrawler)

    fake_robot_hat = types.ModuleType("robot_hat")
    fake_robot_hat.TTS = MagicMock(name="TTS")
    fake_robot_hat.Ultrasonic = MagicMock(name="Ultrasonic")
    fake_robot_hat.Pin = MagicMock(name="Pin")
    monkeypatch.setitem(sys.modules, "robot_hat", fake_robot_hat)
    fake_robot_hat_utils = types.ModuleType("robot_hat.utils")
    fake_robot_hat_utils.enable_speaker = MagicMock(name="enable_speaker")
    monkeypatch.setitem(sys.modules, "robot_hat.utils", fake_robot_hat_utils)
    # Attach submodule as attribute of parent so `import robot_hat.utils`
    # (and attribute access) both work.
    fake_robot_hat.utils = fake_robot_hat_utils

    fake_vilib = types.ModuleType("vilib")
    vilib_class = MagicMock(name="Vilib")
    vilib_class.detect_obj_parameter = {}
    vilib_class.img = None
    fake_vilib.Vilib = vilib_class
    monkeypatch.setitem(sys.modules, "vilib", fake_vilib)

    return fake_picrawler, fake_robot_hat, fake_vilib


@pytest.fixture
def real_hw(monkeypatch):
    picrawler_mod, robot_hat_mod, vilib_mod = _install_fake_sunfounder(monkeypatch)
    from mcp_picrawler.hardware import RealHardware

    hw = RealHardware()
    return hw, picrawler_mod, robot_hat_mod, vilib_mod


# ------------------------------------------------------------ init contract

def test_real_hardware_constructs_picrawler_with_no_args(real_hw):
    _, picrawler_mod, _, _ = real_hw
    picrawler_mod.Picrawler.assert_called_once_with()


def test_real_hardware_wraps_ultrasonic_pins_in_pin_objects(real_hw):
    _, _, robot_hat_mod, _ = real_hw
    robot_hat_mod.Ultrasonic.assert_called_once()
    # Pin("D2") + Pin("D3") — both as positional args
    pin_calls = robot_hat_mod.Pin.call_args_list
    pin_names = [c.args[0] for c in pin_calls]
    assert pin_names == ["D2", "D3"]


def test_real_hardware_starts_camera_and_web_stream(real_hw):
    _, _, _, vilib_mod = real_hw
    vilib_mod.Vilib.camera_start.assert_called_once_with(vflip=False, hflip=False)
    vilib_mod.Vilib.display.assert_called_once_with(local=False, web=True)


def test_real_hardware_calls_enable_speaker(real_hw):
    import robot_hat.utils

    robot_hat.utils.enable_speaker.assert_called_once()


# ------------------------------------------------------------ do_action

def test_do_action_uses_step_not_steps_kwarg(real_hw):
    hw, picrawler_mod, _, _ = real_hw
    picrawler_instance = picrawler_mod.Picrawler.return_value
    hw.do_action("forward", steps=3, speed=70)
    picrawler_instance.do_action.assert_called_once_with("forward", step=3, speed=70)


def test_stop_calls_stand_action(real_hw):
    hw, picrawler_mod, _, _ = real_hw
    picrawler_instance = picrawler_mod.Picrawler.return_value
    hw.stop()
    picrawler_instance.do_action.assert_called_once_with("stand", step=1, speed=80)


# ------------------------------------------------------------ ultrasonic

def test_read_distance_passes_through_float(real_hw):
    hw, _, robot_hat_mod, _ = real_hw
    hw._ultrasonic.read.return_value = 42.5
    assert hw.read_distance_cm() == 42.5


def test_read_distance_preserves_negative_sentinels(real_hw):
    """Ultrasonic uses -1 (timeout) / -2 (failure) — we pass these up, don't mask as 0."""
    hw, _, _, _ = real_hw
    hw._ultrasonic.read.return_value = -1
    assert hw.read_distance_cm() == -1.0
    hw._ultrasonic.read.return_value = -2
    assert hw.read_distance_cm() == -2.0


# ------------------------------------------------------------ frame access

def test_latest_frame_bgr_raises_when_not_ready(real_hw):
    hw, _, _, vilib_mod = real_hw
    vilib_mod.Vilib.img = None
    with pytest.raises(RuntimeError, match="camera frame not ready"):
        hw.latest_frame_bgr(retry_budget_s=0)


def test_latest_frame_bgr_rejects_manager_list_sentinel(real_hw):
    """Vilib.img starts as a Manager().list before the camera thread reassigns it."""
    hw, _, _, vilib_mod = real_hw
    vilib_mod.Vilib.img = [1, 2, 3]  # list, not ndarray → not ready yet
    with pytest.raises(RuntimeError, match="camera frame not ready"):
        hw.latest_frame_bgr(retry_budget_s=0)


def test_latest_frame_bgr_polls_until_ready(real_hw):
    """Simulate the race: list until a deadline, then ndarray."""
    import numpy as np

    hw, _, _, vilib_mod = real_hw
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    call_count = {"n": 0}

    # property getter swaps to a real ndarray on the 3rd access
    def img_getter(self):
        call_count["n"] += 1
        return [1, 2, 3] if call_count["n"] < 3 else frame

    type(vilib_mod.Vilib).img = property(img_getter)
    try:
        got = hw.latest_frame_bgr(retry_budget_s=2.0)
        assert got is frame
        assert call_count["n"] >= 3
    finally:
        del type(vilib_mod.Vilib).img


def test_latest_frame_bgr_returns_ndarray(real_hw):
    import numpy as np

    hw, _, _, vilib_mod = real_hw
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    vilib_mod.Vilib.img = frame
    got = hw.latest_frame_bgr()
    assert got is frame


# ------------------------------------------------------------ vision toggles

def test_set_vision_face_calls_face_detect_switch(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("face", True)
    vilib_mod.Vilib.face_detect_switch.assert_called_once_with(True)


def test_set_vision_qr_calls_qrcode_detect_switch(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("qr", True)
    vilib_mod.Vilib.qrcode_detect_switch.assert_called_once_with(True)


def test_set_vision_gesture_calls_hands_detect_switch(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("gesture", True)
    vilib_mod.Vilib.hands_detect_switch.assert_called_once_with(True)


def test_set_vision_traffic_calls_traffic_detect_switch(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("traffic", True)
    vilib_mod.Vilib.traffic_detect_switch.assert_called_once_with(True)


def test_set_vision_color_enables_via_color_detect(real_hw):
    """color has no *_switch — color_detect(name) enables with target."""
    hw, _, _, vilib_mod = real_hw
    hw.state.target_color = "green"
    hw.set_vision("color", True)
    vilib_mod.Vilib.color_detect.assert_called_once_with("green")


def test_set_vision_color_defaults_to_red_when_no_target_set(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("color", True)
    vilib_mod.Vilib.color_detect.assert_called_once_with("red")


def test_set_vision_color_disable_calls_close_color_detection(real_hw):
    hw, _, _, vilib_mod = real_hw
    hw.set_vision("color", False)
    vilib_mod.Vilib.close_color_detection.assert_called_once_with()


def test_set_vision_rejects_unknown_feature(real_hw):
    hw, _, _, _ = real_hw
    with pytest.raises(ValueError):
        hw.set_vision("xray", True)


def test_object_is_not_in_detections():
    """vilib's object_detect requires model loading — we skip exposing it."""
    from mcp_picrawler.hardware import DETECTIONS

    assert "object" not in DETECTIONS


# ------------------------------------------------------------ set_target_color

def test_set_target_color_does_not_enable_detection(real_hw):
    """Setting a target while detection is OFF must not call color_detect."""
    hw, _, _, vilib_mod = real_hw
    hw.set_target_color("blue")
    assert hw.state.target_color == "blue"
    vilib_mod.Vilib.color_detect.assert_not_called()


def test_set_target_color_updates_active_detection(real_hw):
    """If detection is already on, changing colour retargets live."""
    hw, _, _, vilib_mod = real_hw
    hw.state.target_color = "red"
    hw.set_vision("color", True)  # one call to color_detect("red")
    hw.set_target_color("blue")  # should trigger color_detect("blue")
    calls = [c.args for c in vilib_mod.Vilib.color_detect.call_args_list]
    assert ("red",) in calls
    assert ("blue",) in calls


# ------------------------------------------------------------ read_detections

def test_read_detections_reads_human_n_for_face(real_hw):
    hw, _, _, vilib_mod = real_hw
    vilib_mod.Vilib.detect_obj_parameter.update({"human_n": 2})
    hw.state.vision["face"] = True
    dets = hw.read_detections()
    assert any(d.kind == "face" and d.data["n"] == 2 for d in dets)


def test_read_detections_includes_qr_data(real_hw):
    hw, _, _, vilib_mod = real_hw
    vilib_mod.Vilib.detect_obj_parameter.update({"qr_data": "https://example.com"})
    hw.state.vision["qr"] = True
    dets = hw.read_detections()
    assert any(d.kind == "qr" and d.data["data"] == "https://example.com" for d in dets)


# ------------------------------------------------------------ action list pinning

def test_builtin_actions_pinned_to_known_good_set():
    """Lock in the valid picrawler action names. Drift = breaking change."""
    from mcp_picrawler.hardware import BUILTIN_ACTIONS

    expected = {
        "forward", "backward", "turn left", "turn right",
        "turn left angle", "turn right angle",
        "stand", "sit", "ready",
        "push up", "wave", "dance",
        "look left", "look right", "look up", "look down",
    }
    assert set(BUILTIN_ACTIONS) == expected


def test_no_stale_actions_present():
    """These were in earlier versions but aren't real picrawler motions."""
    from mcp_picrawler.hardware import BUILTIN_ACTIONS

    for stale in ("twist", "beckon", "push-up"):
        assert stale not in BUILTIN_ACTIONS, f"{stale!r} is not a real picrawler action"

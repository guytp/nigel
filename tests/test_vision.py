"""Unit tests for VisionStack — tiers T0 (motion/pHash) and T2 (YOLO).

T3 (caption) needs a big transformers download and is skipped unless
RUN_CAPTION_TESTS=1 is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mcp_picrawler.vision import Detection, ScanResult, VisionStack


@pytest.fixture
def vision():
    return VisionStack()


def _frame(val: int = 30, size=(240, 320, 3)) -> np.ndarray:
    return np.full(size, val, dtype=np.uint8)


def test_scan_result_to_dict_shape(vision):
    r = vision.scan(_frame(), include_objects=False)
    d = r.to_dict()
    assert {"motion", "phash", "objects", "elapsed_ms", "tiers"} <= d.keys()
    assert isinstance(d["motion"], float)
    assert isinstance(d["phash"], str) and len(d["phash"]) == 16
    assert isinstance(d["objects"], list)
    assert "motion" in d["tiers"] and "phash" in d["tiers"]


def test_motion_is_zero_on_first_frame_then_nonzero_on_change(vision):
    r1 = vision.scan(_frame(val=30), include_objects=False)
    assert r1.motion == 0.0  # no prior frame
    r2 = vision.scan(_frame(val=200), include_objects=False)
    assert r2.motion > 0.1  # big grey-level change


def test_motion_is_small_when_frame_barely_changes(vision):
    vision.scan(_frame(val=30), include_objects=False)
    r = vision.scan(_frame(val=31), include_objects=False)
    assert 0 <= r.motion < 0.05


def test_phash_is_stable_for_identical_frame(vision):
    f = _frame(val=50)
    # motion needs two frames to seed state, so use a separate pass
    h1 = vision._phash(f)
    h2 = vision._phash(f)
    assert h1 == h2


def test_phash_differs_for_different_frames(vision):
    gradient_a = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    gradient_b = np.tile(np.linspace(255, 0, 320, dtype=np.uint8), (240, 1))
    fa = np.stack([gradient_a] * 3, axis=-1)
    fb = np.stack([gradient_b] * 3, axis=-1)
    assert vision._phash(fa) != vision._phash(fb)


def test_scan_with_objects_adds_tier_even_on_empty_result(vision):
    # ultralytics should load lazily; a uniform frame has no objects
    r = vision.scan(_frame(), include_objects=True)
    assert "objects" in r.tiers
    assert isinstance(r.objects, list)


def test_detection_dataclass_shape():
    d = Detection(label="cat", conf=0.9, bbox=(1, 2, 3, 4))
    assert d.label == "cat"
    assert d.conf == 0.9
    assert d.bbox == (1, 2, 3, 4)


@pytest.mark.skipif(
    os.environ.get("RUN_CAPTION_TESTS") != "1",
    reason="caption tier pulls ~2GB of model weights; enable with RUN_CAPTION_TESTS=1",
)
def test_caption_returns_string(vision):
    text = vision.caption(_frame())
    assert isinstance(text, str) and len(text) > 0

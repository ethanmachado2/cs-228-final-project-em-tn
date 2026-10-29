"""SBMEX extraction against the real pyfing stack. Run in the container (Req 6.1, 5.8)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from fpv import config as C
from fpv.extract import ExtractionResult, extract
from fpv.imaging import load_gray8
from fpv.preprocess import preprocess

pytestmark = pytest.mark.toolchain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_SAMPLE = PROJECT_ROOT / "data/raw/Real/100__M_Left_index_finger.BMP"


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT)


PERIOD = 9                                   # 500 ppi-like ridge spacing
H, W, GAP = 240, 220, 16
BREAKS = ((W // 2 - 3 * PERIOD + 4.5, H // 2 - 40), (W // 2 + 4.5, H // 2), (W // 2 + 3 * PERIOD + 4.5, H // 2 + 40))


@pytest.fixture(scope="module")
def broken_grating() -> np.ndarray:
    """Vertical dark ridges in an ellipse with three known breaks: a GAP-px segment of one
    ridge is painted valley-bright at each point in BREAKS, giving two ridge endings each."""
    y, x = np.mgrid[:H, :W].astype(np.float64)
    img = 127.5 + 110 * np.cos(2 * np.pi * x / PERIOD)   # ridge centres at x = 4.5 + 9k
    for cx, cy in BREAKS:
        img[(np.abs(x - cx) <= PERIOD / 2) & (np.abs(y - cy) < GAP / 2)] = 237.5
    inside = ((x - W / 2) / (W / 2 - 8)) ** 2 + ((y - H / 2) / (H / 2 - 8)) ** 2 <= 1
    return np.where(inside, img, 235).astype(np.uint8)


@pytest.fixture(scope="module")
def native_attempt(conf):
    """Attempt 1 without the SOCOFing 3x resample: the grating already has a 500 ppi period."""
    return dataclasses.replace(conf.attempt(1), resample=1.0)


@pytest.fixture(scope="module")
def pre(conf, broken_grating, native_attempt):
    return preprocess(broken_grating, native_attempt, configuration_id=conf.configuration_id(native_attempt))


def test_minutiae_lie_inside_mask_with_valid_directions(pre):
    ex = extract(pre)
    assert isinstance(ex, ExtractionResult)
    assert ex.count == len(ex.minutiae) > 0
    assert ex.configuration_id == pre.configuration_id
    h, w = pre.mask.shape
    for m in ex.minutiae:
        assert 0 <= m.x < w and 0 <= m.y < h
        assert pre.mask[m.y, m.x] == 255, m
        assert -math.pi <= m.direction <= math.pi
        assert isinstance(m.quality, float) and m.type in ("E", "B")
    # Every known break is found: some minutia within one ridge period of each break.
    for cx, cy in BREAKS:
        assert any(abs(m.x - cx) <= PERIOD and abs(m.y - cy) <= GAP for m in ex.minutiae), (cx, cy)


def test_two_runs_give_identical_minutiae(conf, broken_grating, pre, native_attempt):
    a = native_attempt
    again = preprocess(broken_grating.copy(), a, configuration_id=conf.configuration_id(a))
    assert extract(pre).minutiae == extract(again).minutiae == extract(pre).minutiae


@pytest.mark.dataset
def test_real_sample_yields_minutiae_on_some_attempt(conf):
    img = load_gray8(REAL_SAMPLE)
    counts = [extract(preprocess(img, a, configuration_id=conf.configuration_id(a))).count
              for a in conf.attempts]
    assert max(counts) > 0, counts

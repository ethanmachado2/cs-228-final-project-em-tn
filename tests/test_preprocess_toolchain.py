"""pyfing preprocessing chain against the real libraries. Run in the container (Req 5.5, 5.8, 5.10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpv import config as C
from fpv.cli import main
from fpv.imaging import load_gray8, scaled_size
from fpv.preprocess import preprocess, quality_after, quality_after_image
from fpv.quality import QualityResult, QualitySettings

pytestmark = pytest.mark.toolchain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = (103, 96)  # SOCOFing size
MEDIUM_SAMPLE = "data/raw/Altered/Altered-Medium/100__M_Left_index_finger_CR.BMP"


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def grating() -> np.ndarray:
    """Synthetic oblique sinusoidal ridge pattern (period 9 px) inside an elliptical finger."""
    y, x = np.mgrid[: ORIGINAL[0], : ORIGINAL[1]].astype(np.float64)
    theta = 0.5
    ridges = 127.5 + 100 * np.sin(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / 9)
    inside = ((x - 48) / 40) ** 2 + ((y - 51) / 46) ** 2 <= 1
    return np.where(inside, ridges, 230).astype(np.uint8)


@pytest.fixture(scope="module")
def results(conf, grating):
    return {a.attempt_no: preprocess(grating, a, configuration_id=conf.configuration_id(a))
            for a in conf.attempts}


def test_every_attempt_returns_upscaled_uint8_outputs(conf, results):
    for a in conf.attempts:
        r = results[a.attempt_no]
        expected = scaled_size(scaled_size(ORIGINAL, a.resample), a.upscale)
        assert r.enhanced.shape == expected and r.mask.shape == expected, a.name
        assert r.enhanced.dtype == np.uint8 and r.mask.dtype == np.uint8
        assert set(np.unique(r.mask).tolist()) <= {0, 255}
        assert r.scale_factor == a.scale_factor and r.original_shape == ORIGINAL
        assert r.configuration_id == conf.configuration_id(a)
        assert set(r.timings_ms) == {"segmentation", "orientation", "frequency", "enhancement"}


def test_repeated_calls_are_byte_identical(conf, grating, results):
    for a in conf.attempts:
        again = preprocess(grating.copy(), a, configuration_id=conf.configuration_id(a))
        first = results[a.attempt_no]
        assert np.array_equal(first.enhanced, again.enhanced), a.name
        assert np.array_equal(first.mask, again.mask), a.name
        assert first.enhanced.tobytes() == again.enhanced.tobytes()


def test_quality_after_scores_500ppi_image(conf, results):
    settings = QualitySettings.from_config(conf)
    for a in conf.attempts:
        r = results[a.attempt_no]
        assert quality_after_image(r).shape == ORIGINAL
        q = quality_after(r, settings)
        assert isinstance(q, QualityResult)
        assert q.score is None or 0 <= q.score <= 100
        assert q.upscale == settings.upscale
        assert q.scored_shape == scaled_size(ORIGINAL, settings.upscale)  # 500 ppi size x NFIQ 2 upscale


@pytest.mark.dataset
def test_cli_preprocess_medium_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    out_png = tmp_path / "enh.png"
    lines: list[str] = []
    code = main(["preprocess", MEDIUM_SAMPLE, "--attempt", "2", "--out", str(out_png)],
                out=lines.append, err=lambda s: None)
    text = "\n".join(lines)
    assert code == 0, text
    a2 = C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT).attempt(2)
    assert load_gray8(out_png).shape == scaled_size(scaled_size(ORIGINAL, a2.resample), a2.upscale)
    assert "quality_after" in text and "configuration_id  cfg_" in text

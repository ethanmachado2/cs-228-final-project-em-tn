"""NFIQ 2 quality stage against the real binary. Run in the container (Req 4.3, 4.5, 4.8)."""

from __future__ import annotations

import numpy as np
import pytest

from fpv import config as C
from fpv.cli import main
from fpv.imaging import load_gray8
from fpv.quality import QualitySettings, assess_quality, quality_gate

pytestmark = pytest.mark.toolchain

REAL_SAMPLE = "data/raw/Real/100__M_Left_index_finger.BMP"


@pytest.fixture
def settings(default_config_path, project_root):
    return QualitySettings.from_config(C.load(default_config_path, root=project_root))


def test_blank_image_scores_none(settings):
    r = assess_quality(np.full((103, 96), 255, np.uint8), settings)
    assert r.score is None
    assert "blank" in r.error
    assert quality_gate(r, 5).reason == "nfiq2_failed"


@pytest.mark.dataset
def test_real_sample_scores_with_feedback(settings, project_root):
    r = assess_quality(load_gray8(project_root / REAL_SAMPLE), settings)
    assert r.score is not None and 0 <= r.score <= 100
    assert r.actionable_feedback
    assert r.scored_shape == (309, 288)


@pytest.mark.dataset
def test_cli_quality_real_sample(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    lines: list[str] = []
    code = main(["quality", REAL_SAMPLE], out=lines.append, err=lambda s: None)
    text = "\n".join(lines)
    assert "feedback" in text and ("PASS" in text or "FAIL (below_threshold)" in text)
    assert code in (0, 20)

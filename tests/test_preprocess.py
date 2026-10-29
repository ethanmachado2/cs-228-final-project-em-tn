"""Unit tests for preprocess.py that do not need pyfing (quality_after plumbing, CLI input errors)."""

from __future__ import annotations

import numpy as np
import pytest

from fpv import config as C
from fpv.cli import main
from fpv.imaging import scaled_size
from fpv.preprocess import PreprocessResult, quality_after, quality_after_image
from fpv.quality import QualityResult, QualitySettings


def _result(scale: float, original=(103, 96)) -> PreprocessResult:
    h, w = scaled_size(original, scale)
    xs = np.arange(w) * original[1] / w          # original-scale x of each upscaled column
    enhanced = np.zeros((h, w), np.uint8)
    enhanced[:, (xs // 4) % 2 == 0] = 255        # 4 px wide white ridges at 500 ppi
    mask = np.zeros((h, w), np.uint8)
    mask[: h // 2] = 255                         # top half is foreground
    attempt = C.AttemptConfig(3, "deep_boost", "SUFS", "SNFOE", "SNFFE", "SNFEN", upscale=scale)
    return PreprocessResult(attempt, enhanced, mask, scale, original, "cfg_x", {})


@pytest.mark.parametrize("scale", [1.0, 2.0, 1.5, 3.0])
def test_quality_after_image_is_original_size_inverted_and_masked(scale):
    r = _result(scale)
    img = quality_after_image(r)
    assert img.shape == r.original_shape and img.dtype == np.uint8
    h = r.original_shape[0]
    assert (img[h // 2 + 2 :] == 255).all()      # background -> white
    assert img[: h // 2 - 2].min() < 64          # ridges inverted to dark


def test_quality_after_uses_given_settings():
    r = _result(2.0)
    settings = QualitySettings(upscale=3.0)
    calls = []

    def fake_assess(image, s):
        calls.append((image.shape, s))
        return QualityResult(None, {}, "", "t", 0.0, upscale=s.upscale, error="x",
                             scored_shape=scaled_size(image.shape, s.upscale))

    q = quality_after(r, settings, assess=fake_assess)
    assert calls == [((103, 96), settings)]
    assert q.score is None and q.scored_shape == (309, 288)


def test_cli_preprocess_rejects_bad_inputs(tmp_path, project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    errors: list[str] = []
    img = tmp_path / "x.png"
    img.write_bytes(b"not an image")
    assert main(["preprocess", str(img), "--attempt", "1", "--out", str(tmp_path / "o.png")],
                out=lambda s: None, err=errors.append) == 2
    assert "cannot decode" in errors[-1]
    import cv2

    cv2.imwrite(str(img), np.full((20, 20), 128, np.uint8))
    assert main(["preprocess", str(img), "--attempt", "4", "--out", str(tmp_path / "o.png")],
                out=lambda s: None, err=errors.append) == 2
    assert "attempt 4 is not defined" in errors[-1]
    assert main(["preprocess", str(img), "--attempt", "1", "--out", str(tmp_path / "nodir" / "o.png")],
                out=lambda s: None, err=errors.append) == 2
    assert "output directory" in errors[-1]

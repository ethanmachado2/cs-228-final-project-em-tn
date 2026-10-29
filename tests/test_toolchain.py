"""Toolchain smoke tests. Run in the container: `docker compose run --rm fpv pytest -m toolchain`."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import FIXTURES

pytestmark = pytest.mark.toolchain

BZ = FIXTURES / "bozorth3"


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, **kw)


REAL_SAMPLE = "data/raw/Real/100__M_Left_index_finger.BMP"


def _nfiq2_csv(png):
    """Run `nfiq2 -F -a <png>` and return (header fields, row fields) of the actionable CSV."""
    proc = _run(["nfiq2", "-F", "-a", str(png)])
    assert proc.returncode == 0  # NFIQ 2 exits 0 even when it cannot score; errors are in the CSV
    header, row = proc.stdout.strip().splitlines()[-2:]
    return header.split(","), row.split(",")


def _socofing_png(project_root, tmp_path, scale):
    import cv2
    from PIL import Image

    img = cv2.imread(str(project_root / REAL_SAMPLE), cv2.IMREAD_GRAYSCALE)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = tmp_path / f"real_x{scale}.png"
    Image.fromarray(img).save(out, dpi=(500, 500))
    return out


def test_nfiq2_runs_and_reports_pinned_version():
    proc = _run(["nfiq2", "-h"])
    assert "NFIQ 2: 2.3.0" in proc.stdout + proc.stderr


def test_nfiq2_blank_image_reports_error_without_crashing(tmp_path):
    from PIL import Image

    png = tmp_path / "blank.png"
    Image.fromarray(np.full((103, 96), 255, np.uint8)).save(png, dpi=(500, 500))
    header, row = _nfiq2_csv(png)
    assert header[:4] == ['"Filename"', "FingerCode", "QualityScore", '"OptionalError"']
    assert row[2] == "NA" and "blank" in row[3]


@pytest.mark.dataset
def test_nfiq2_scores_socofing_sample(project_root, tmp_path):
    # Spike finding: native-size SOCOFing images (96x103) cannot be scored (FDA error);
    # a 2x upscale yields a score. The gate policy for this is an open decision for Task 3.
    _, native = _nfiq2_csv(_socofing_png(project_root, tmp_path, 1))
    assert native[2] == "NA" and "Frequency Domain Analysis" in native[3]
    header, row = _nfiq2_csv(_socofing_png(project_root, tmp_path, 2))
    assert "UniformImage" in header and "SufficientFingerprintForeground" in header
    assert row[2].isdigit() and 0 <= int(row[2]) <= 100


def test_bozorth3_prints_usage():
    proc = _run(["bozorth3", "-h"])
    assert proc.returncode == 0
    assert "Usage" in proc.stdout + proc.stderr


def test_bozorth3_default_xyt_self_match():
    tpl = BZ / "handwritten_12.xyt"
    self_score = int(_run(["bozorth3", str(tpl), str(tpl)]).stdout.strip())
    other = int(_run(["bozorth3", str(tpl), str(BZ / "unrelated_12.xyt")]).stdout.strip())
    assert self_score >= 40 > other


def test_import_pyfing():
    import pyfing

    assert pyfing.__version__ == "0.7.3"


def test_cv_ximgproc_available():
    import cv2

    assert hasattr(cv2, "ximgproc")
    img = np.zeros((32, 32), np.uint8)
    img[10:20, 5:27] = 255
    assert cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL).shape == img.shape


def test_sufs_weights_load():
    import pyfing

    sufs = pyfing.Sufs()
    assert sufs.model.count_params() > 0
    assert any(np.any(w) for w in sufs.model.get_weights())


def test_fpv_doctor_exits_zero(project_root):
    proc = _run([sys.executable, "-m", "fpv.cli", "doctor"], cwd=project_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in ("nfiq2", "nbis/bozorth3", "pyfing", "tensorflow", "keras", "opencv", "cv.ximgproc"):
        assert name in proc.stdout
    assert "all checks passed" in proc.stdout

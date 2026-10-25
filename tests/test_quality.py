"""Unit tests for fpv.quality with a mocked NFIQ 2 runner (Req 4.1-4.6, 4.8)."""

from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pytest

from conftest import FIXTURES
from fpv import config as C
from fpv import quality as Q
from fpv.cli import main

NFIQ2 = FIXTURES / "nfiq2"


def _stdout(case: str) -> str:
    return (NFIQ2 / f"{case}.stdout").read_text()


class FakeRunner:
    """Records each call and inspects the PNG while it still exists."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "", exc: Exception | None = None):
        self.stdout, self.returncode, self.stderr, self.exc = stdout, returncode, stderr, exc
        self.calls: list[tuple[tuple, dict]] = []
        self.png_shapes: list[tuple[int, ...]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        png = args[0][-1]
        self.png_shapes.append(cv2.imread(png, cv2.IMREAD_UNCHANGED).shape)
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(args[0], self.returncode, self.stdout, self.stderr)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(103, 96), dtype=np.uint8)


SETTINGS = Q.QualitySettings(upscale=3.0, tool_version="2.3.0")


# ------------------------------------------------------------------ parsing


def test_parse_normal_output():
    p = Q.parse_nfiq2_output(_stdout("score_actionable"))
    assert p.score == 34
    assert p.error is None
    assert p.actionable_feedback == {
        "UniformImage": 97.4563,
        "EmptyImageOrContrastTooLow": 130.65,
        "FingerprintImageWithMinutiae": 50.0,
        "SufficientFingerprintForeground": 28200.0,
    }


def test_parse_verbose_output_keeps_actionable_only():
    p = Q.parse_nfiq2_output(_stdout("score_actionable_verbose"))
    assert p.score == 34
    assert set(p.actionable_feedback) == set(Q.ACTIONABLE_FIELDS)


@pytest.mark.parametrize(
    ("case", "needle"),
    [
        ("fail_blank", "All image rows appear to be blank"),
        ("fail_native_socofing", "Frequency Domain Analysis"),
        ("fail_missing_file", "ObjectDoesNotExist"),
    ],
)
def test_parse_failure_output_gives_none(case, needle):
    p = Q.parse_nfiq2_output(_stdout(case))
    assert p.score is None
    assert p.actionable_feedback == {}
    assert needle in p.error


@pytest.mark.parametrize("case", ["score_plain", "fail_plain_native", "no_args"])
def test_parse_rejects_non_csv_output(case):
    with pytest.raises(Q.QualityToolError):
        Q.parse_nfiq2_output(_stdout(case))


def test_parse_rejects_out_of_range_score():
    bad = _stdout("score_actionable").replace(",0,34,NA,", ",0,140,NA,")
    with pytest.raises(Q.QualityToolError, match="outside"):
        Q.parse_nfiq2_output(bad)


# ------------------------------------------------------------------ assess_quality


def test_assess_quality_normal(image):
    runner = FakeRunner(_stdout("score_actionable"))
    r = Q.assess_quality(image, SETTINGS, runner=runner)
    assert r.score == 34 and r.error is None
    assert r.actionable_feedback["FingerprintImageWithMinutiae"] == 50.0
    assert r.tool_version == "2.3.0" and r.upscale == 3.0 and r.duration_ms >= 0
    assert r.raw_output == _stdout("score_actionable")


def test_assess_quality_failure_output_gives_none(image):
    r = Q.assess_quality(image, SETTINGS, runner=FakeRunner(_stdout("fail_native_socofing")))
    assert r.score is None and "FDA" in r.error


def test_runner_gets_argument_list_without_shell(image):
    runner = FakeRunner(_stdout("score_actionable"))
    Q.assess_quality(image, SETTINGS, runner=runner)
    (args, kwargs), = runner.calls
    cmd = args[0]
    assert isinstance(cmd, list) and all(isinstance(a, str) for a in cmd)
    assert cmd[:3] == ["nfiq2", "-F", "-a"] and cmd[3].endswith(".png")
    assert kwargs.get("shell", False) is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["timeout"] == SETTINGS.timeout_s


@pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
def test_upscale_factor_is_applied(image, factor):
    runner = FakeRunner(_stdout("score_actionable"))
    r = Q.assess_quality(image, Q.QualitySettings(upscale=factor), runner=runner)
    expected = (int(103 * factor), int(96 * factor))
    assert runner.png_shapes == [expected]  # 2-D => 8-bit grayscale PNG
    assert r.scored_shape == expected


def test_upscale_uses_config_value(image, default_config_path, project_root):
    conf = C.load(default_config_path, root=project_root)
    settings = Q.QualitySettings.from_config(conf)
    assert settings.upscale == conf.nfiq2_upscale == 3.0
    assert settings.tool_version == conf.tools["nfiq2"]
    runner = FakeRunner(_stdout("score_actionable"))
    Q.assess_quality(image, settings, runner=runner)
    assert runner.png_shapes == [(309, 288)]


@pytest.mark.parametrize(
    "exc",
    [FileNotFoundError("nfiq2"), subprocess.TimeoutExpired(["nfiq2"], 30), PermissionError("denied")],
    ids=["missing", "timeout", "oserror"],
)
def test_tool_errors_raise(image, exc):
    with pytest.raises(Q.QualityToolError):
        Q.assess_quality(image, SETTINGS, runner=FakeRunner(exc=exc))


@pytest.mark.parametrize("code", [1, -9])
def test_nonzero_exit_raises(image, code):
    runner = FakeRunner(_stdout("score_actionable"), returncode=code, stderr="boom")
    with pytest.raises(Q.QualityToolError, match="signal 9" if code < 0 else "exit code 1"):
        Q.assess_quality(image, SETTINGS, runner=runner)


def test_settings_validation():
    with pytest.raises(ValueError):
        Q.QualitySettings(upscale=0)
    with pytest.raises(ValueError):
        Q.QualitySettings(upscale=2.0, timeout_s=0)


# ------------------------------------------------------------------ gate


def _result(score):
    return Q.QualityResult(score=score, actionable_feedback={}, raw_output="", tool_version="2.3.0",
                           duration_ms=1.0)


def test_gate_passes_at_threshold():
    d = Q.quality_gate(_result(5), 5)
    assert d == Q.GateDecision(True, 5, 5, None)


def test_gate_fails_below_threshold():
    d = Q.quality_gate(_result(4), 5)
    assert d == Q.GateDecision(False, 4, 5, "below_threshold")


def test_gate_fails_on_none():
    d = Q.quality_gate(_result(None), 5)
    assert d == Q.GateDecision(False, None, 5, "nfiq2_failed")


# ------------------------------------------------------------------ CLI


def _cli(monkeypatch, tmp_path, stdout, project_root):
    png = tmp_path / "img.png"
    cv2.imwrite(str(png), np.full((103, 96), 128, np.uint8))
    runner = FakeRunner(stdout)
    real = Q.assess_quality
    monkeypatch.setattr(Q, "assess_quality", lambda img, s: real(img, s, runner=runner))
    monkeypatch.chdir(project_root)
    lines: list[str] = []
    code = main(["quality", str(png)], out=lines.append, err=lambda s: None)
    return code, "\n".join(lines)


def test_cli_quality_pass(monkeypatch, tmp_path, project_root):
    code, text = _cli(monkeypatch, tmp_path, _stdout("score_actionable"), project_root)
    assert code == 0
    assert "score     34" in text and "FingerprintImageWithMinutiae" in text and "PASS" in text
    assert "288x309" in text and "upscale 3x" in text


def test_cli_quality_fail_nfiq2_failed(monkeypatch, tmp_path, project_root):
    code, text = _cli(monkeypatch, tmp_path, _stdout("fail_blank"), project_root)
    assert code == 20
    assert "score     NA" in text and "FAIL (nfiq2_failed)" in text


def test_cli_quality_unreadable_image(tmp_path, monkeypatch, project_root):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    monkeypatch.chdir(project_root)
    errs: list[str] = []
    assert main(["quality", str(bad)], out=lambda s: None, err=errs.append) == 2
    assert "cannot decode" in errs[-1]

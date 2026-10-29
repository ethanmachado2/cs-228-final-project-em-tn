"""Unit tests for `fpv.matcher` with an injected subprocess runner (Req 9.3-9.7, 9.11)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fpv.adapter import XytMinutia, XytTemplate, write_xyt
from fpv.cli import EXIT_ERROR, EXIT_REJECTED_MINUTIAE, EXIT_USAGE, main
from fpv.matcher import (
    MIN_BOZORTH_MINUTIAE,
    InsufficientMinutiae,
    MatcherError,
    MatchResult,
    decide,
    match,
    parse_bozorth3_output,
)
from fpv.types import Status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"


def _template(n: int, offset: int = 0) -> XytTemplate:
    return XytTemplate(tuple(XytMinutia(x=5 + 7 * i, y=3 + offset + 5 * i, theta=(37 * i) % 360,
                                        quality=50) for i in range(n)))


@pytest.fixture
def pair(tmp_path):
    def make(n_probe: int = 12, n_gallery: int = 12) -> tuple[Path, Path]:
        probe, gallery = tmp_path / "probe.xyt", tmp_path / "gallery.xyt"
        write_xyt(_template(n_probe), probe)
        write_xyt(_template(n_gallery, offset=1), gallery)
        return probe, gallery
    return make


class FakeRunner:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "",
                 exc: BaseException | None = None) -> None:
        self.stdout, self.returncode, self.stderr, self.exc = stdout, returncode, stderr, exc
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def test_parses_score_and_invokes_argument_list(pair):
    probe, gallery = pair()
    runner = FakeRunner(stdout="598\n")
    r = match(probe, gallery, bozorth3_bin="/opt/bz3", timeout_s=7, tool_version="5.0.0", runner=runner)
    assert isinstance(r, MatchResult)
    assert (r.score, r.probe_count, r.gallery_count, r.tool_version) == (598, 12, 12, "5.0.0")
    assert r.duration_ms >= 0
    (cmd, kwargs), = runner.calls
    assert cmd == ["/opt/bz3", str(probe), str(gallery)]          # Req 9.7: no shell, no -m1
    assert kwargs["timeout"] == 7 and not kwargs.get("shell", False)
    assert kwargs["stdin"] is subprocess.DEVNULL


@pytest.mark.parametrize("runner, needle", [
    (FakeRunner(returncode=1, stderr="ERROR: fopen() of xyt file failed"), "exit code 1"),
    (FakeRunner(returncode=-11), "signal 11"),
    (FakeRunner(exc=subprocess.TimeoutExpired(["bozorth3"], 30)), "timed out"),
    (FakeRunner(exc=FileNotFoundError("bozorth3")), "not found"),
    (FakeRunner(exc=PermissionError("denied")), "cannot run"),
    (FakeRunner(stdout="abc\n"), "unparseable"),
    (FakeRunner(stdout=""), "unparseable"),
    (FakeRunner(stdout="12 34\n"), "unparseable"),
    (FakeRunner(stdout="-3\n"), "unparseable"),
])
def test_tool_failures_raise_matcher_error(pair, runner, needle):
    probe, gallery = pair()
    with pytest.raises(MatcherError, match=needle):
        match(probe, gallery, runner=runner)


@pytest.mark.parametrize("n_probe, n_gallery, which, count", [
    (9, 12, "probe", 9),
    (12, 9, "gallery", 9),
    (0, 12, "probe", 0),
    (5, 5, "probe", 5),       # probe is checked first
])
def test_insufficient_minutiae_skips_bozorth3(pair, n_probe, n_gallery, which, count):
    probe, gallery = pair(n_probe, n_gallery)
    runner = FakeRunner(stdout="100\n")
    with pytest.raises(InsufficientMinutiae) as ei:
        match(probe, gallery, runner=runner)
    assert (ei.value.which, ei.value.count, ei.value.minimum) == (which, count, MIN_BOZORTH_MINUTIAE)
    assert runner.calls == []                                      # Req 9.5: never invoked


def test_exactly_ten_minutiae_is_enough(pair):
    probe, gallery = pair(10, 10)
    assert match(probe, gallery, runner=FakeRunner(stdout="0\n")).score == 0


def test_parse_bozorth3_output_tolerates_whitespace():
    assert parse_bozorth3_output("  42 \n") == 42


@pytest.mark.parametrize("threshold", [0, 1, 40, 1000])
def test_decide_boundary(threshold):
    assert decide(threshold, threshold) is Status.MATCH
    if threshold > 0:
        assert decide(threshold - 1, threshold) is Status.NO_MATCH
    assert decide(threshold + 1, threshold) is Status.MATCH


# --------------------------------------------------------------------------- CLI (no Bozorth3 call)


def _cli(*argv: str) -> tuple[int, str]:
    lines: list[str] = []
    code = main(["--config", str(CONFIG), "match", *argv], out=lines.append, err=lines.append)
    return code, "\n".join(lines)


def test_cli_short_probe_is_rejected_minutiae(pair):
    probe, gallery = pair(9, 12)
    code, text = _cli(str(probe), str(gallery))
    assert code == EXIT_REJECTED_MINUTIAE and "REJECTED_MINUTIAE" in text


def test_cli_short_gallery_is_error(pair):
    probe, gallery = pair(12, 9)
    code, text = _cli(str(probe), str(gallery))
    assert code == EXIT_ERROR and "ERROR" in text and "gallery" in text


def test_cli_missing_or_corrupt_template_is_usage_error(pair, tmp_path):
    probe, _ = pair()
    assert _cli(str(probe), str(tmp_path / "nope.xyt"))[0] == EXIT_USAGE
    bad = tmp_path / "bad.xyt"
    bad.write_text("1 2 3\n")
    assert _cli(str(probe), str(bad))[0] == EXIT_USAGE

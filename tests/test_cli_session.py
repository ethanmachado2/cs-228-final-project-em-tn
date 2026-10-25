"""Unit tests for `fpv enroll` / `fpv verify` (Task 10, Req 3, 8.7, 9.2, 9.4, 10).

The session loop is driven with scripted input and a fake controller injected through
`main(..., input_fn=..., controller_factory=...)`; one test uses the real controller with fake
stages and a temporary store to check the persisted runs. No toolchain needed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fpv import cli
from fpv.catalog import Catalog
from fpv.pipeline import EnrollmentOutcome, PipelineController, VerificationOutcome
from fpv.quality import QualityResult
from fpv.storage import Repository
from fpv.types import Identity, Status

from test_pipeline import FakeStages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"
ID5 = Identity.parse(5, "Left", "index")
CLAIM = ["5", "left", "index"]


@pytest.fixture
def images(tmp_path: Path) -> SimpleNamespace:
    img = (np.indices((103, 96)).sum(axis=0) % 7 * 30).astype(np.uint8)
    paths = {}
    for name in ("a", "b", "c"):
        p = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(p), img)
        paths[name] = p
    broken = tmp_path / "broken.bmp"
    broken.write_bytes(b"not an image")
    return SimpleNamespace(**paths, broken=broken, missing=tmp_path / "missing.bmp")


def outcome(status: Status, message: str, *, reason: str | None = None, version: int | None = None,
            score: int | None = None, verify: bool = False):
    if verify:
        return VerificationOutcome("run", status, reason, message, 40, (), None, score, 40, None)
    return EnrollmentOutcome("run", status, reason, message, 40, (), "tid" if version else None, version)


REJ_Q = outcome(Status.REJECTED_QUALITY, "Sample rejected: NFIQ 2 quality 3 is below the minimum 5; "
                                         "please present a new sample", reason="below_threshold")
REJ_M = outcome(Status.REJECTED_MINUTIAE, "Sample rejected: fewer than 12 minutiae after 3 attempt(s); "
                                          "please present a new sample", reason="insufficient_minutiae")
OK = outcome(Status.ENROLLED, "Enrollment successful (template v1)", version=1)


class FakeController:
    """Returns scripted outcomes and records every call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, Path, Identity, str, int]] = []
        self.opened = False

    def _next(self, op, path, identity, session_id, presentation_no):
        self.calls.append((op, Path(path), identity, session_id, presentation_no))
        return self.outcomes.pop(0)

    def enroll(self, path, identity, *, session_id, presentation_no):
        return self._next("enroll", path, identity, session_id, presentation_no)

    def verify(self, path, identity, *, session_id, presentation_no):
        return self._next("verify", path, identity, session_id, presentation_no)

    def factory(self, conf):
        self.opened = True
        return contextlib.nullcontext(self)


def run_cli(argv, inputs=(), controller=None):
    """Run `fpv --config <default> <argv>`; every scripted input must be consumed."""
    controller = controller or FakeController([])
    queue = list(inputs)
    prompts: list[str] = []
    out: list[str] = []
    err: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    code = cli.main(["--config", str(CONFIG), *argv], out=out.append, err=err.append,
                    input_fn=input_fn, controller_factory=getattr(controller, "factory", controller))
    assert not queue, f"unconsumed inputs: {queue}"
    return SimpleNamespace(code=code, out=out, err=err, prompts=prompts, ctl=controller)


def test_interactive_reprompts_every_invalid_input(images):
    ctl = FakeController([OK])
    r = run_cli(["enroll"], [str(images.missing), str(images.broken), f'"{images.a}"',
                             "0", "abc", "601", "5",
                             "up", "LEFT",
                             "pinky", "Index",
                             "y"], ctl)
    assert r.code == cli.EXIT_OK
    assert ctl.calls == [("enroll", images.a, ID5, ctl.calls[0][3], 1)]
    text = "\n".join(r.out)
    assert f"no such file: {images.missing}" in text
    assert "cannot decode image" in text
    assert text.count("allowed values: integers 1-600") == 3
    assert "allowed values: Left, Right" in text
    assert "allowed values: thumb, index, middle, ring, little" in text
    assert r.prompts.count("Image path: ") == 3
    assert r.prompts[-1] == "Proceed? [y/N]: "
    assert f"Claim: enroll {ID5} with image {images.a}" in r.out
    assert r.out[-1] == "Enrollment successful (template v1)"
    assert any("config" in line and "cs_" in line for line in r.err)  # Req 3.8


@pytest.mark.parametrize("answer", ["n", "", "maybe"])
def test_declining_confirmation_processes_nothing(images, answer):
    ctl = FakeController([OK])
    r = run_cli(["verify"], [str(images.a), *CLAIM, answer], ctl)
    assert r.code == cli.EXIT_USAGE
    assert ctl.calls == [] and not ctl.opened
    assert r.out[-1] == "Cancelled; no sample was processed."


def test_two_rejections_then_success_is_one_session(images):
    ctl = FakeController([REJ_Q, REJ_M, OK])
    r = run_cli(["enroll"], [str(images.a), *CLAIM, "y", str(images.missing), str(images.b), str(images.c)], ctl)
    assert r.code == cli.EXIT_OK
    assert [(c[1], c[2], c[4]) for c in ctl.calls] == [(images.a, ID5, 1), (images.b, ID5, 2), (images.c, ID5, 3)]
    assert len({c[3] for c in ctl.calls}) == 1
    assert REJ_Q.message in r.out and REJ_M.message in r.out
    assert r.prompts.count("New image path: ") == 3  # the missing path is re-asked
    assert r.out[-1] == "Enrollment successful (template v1)"


def test_three_rejections_reach_the_limit(images):
    ctl = FakeController([REJ_Q, REJ_M, REJ_Q])
    r = run_cli(["enroll"], [str(images.a), *CLAIM, "y", str(images.b), str(images.c)], ctl)
    assert r.code == cli.EXIT_LIMIT_REACHED == 22
    assert len(ctl.calls) == 3
    assert r.out[-1] == "Presentation limit reached (3 of 3); session ended."
    assert r.prompts.count("New image path: ") == 2  # no prompt after the last presentation


@pytest.mark.parametrize("op, result, code", [
    ("enroll", OK, 0),
    ("enroll", outcome(Status.ERROR, "NFIQ 2 failed: boom", reason="nfiq2_tool_error"), 1),
    ("verify", outcome(Status.MATCH, "Match", score=55, verify=True), 0),
    ("verify", outcome(Status.NO_MATCH, "No match", score=12, verify=True), 10),
    ("verify", outcome(Status.NOT_ENROLLED, "Not enrolled", verify=True), 11),
    ("verify", REJ_Q, 20),
    ("verify", REJ_M, 21),
])
def test_terminal_status_message_and_exit_code(images, op, result, code):
    ctl = FakeController([result])
    r = run_cli([op, "--image", str(images.a), "--user", "5", "--hand", "Left", "--finger", "index", "--yes"],
                controller=ctl)
    assert r.code == code
    assert r.prompts == []
    assert len(ctl.calls) == 1 and ctl.calls[0][0] == op
    assert r.out[-1] == result.message
    assert f"run run: {result.status.value}" in "\n".join(r.out)
    if getattr(result, "score", None) is not None:
        assert f"score {result.score} (bozorth3_match=40)" in r.out


def test_exit_codes_are_distinct_per_final_status():
    codes = {s: c for s, c in cli.STATUS_EXIT_CODES.items() if s not in ("ENROLLED", "MATCH")}
    assert len(set(codes.values()) | {cli.EXIT_OK, cli.EXIT_LIMIT_REACHED}) == len(codes) + 2
    assert set(cli.STATUS_EXIT_CODES) == {s.value for s in Status}


@pytest.mark.parametrize("override", [
    {"--image": "MISSING"}, {"--image": "BROKEN"}, {"--user": "0"}, {"--user": "x"},
    {"--hand": "up"}, {"--finger": "pinky"},
])
def test_non_interactive_invalid_input_exits_2_without_processing(images, override):
    flags = {"--image": str(images.a), "--user": "5", "--hand": "Left", "--finger": "index"}
    flags.update({k: {"MISSING": str(images.missing), "BROKEN": str(images.broken)}.get(v, v)
                  for k, v in override.items()})
    ctl = FakeController([OK])
    r = run_cli(["enroll", *[x for kv in flags.items() for x in kv], "--yes"], controller=ctl)
    assert r.code == cli.EXIT_USAGE
    assert r.prompts == [] and ctl.calls == [] and not ctl.opened
    assert r.err[-1].startswith("fpv enroll: ")


def test_non_interactive_rejection_does_not_prompt(images):
    ctl = FakeController([REJ_M])
    r = run_cli(["verify", "--image", str(images.a), "--user", "5", "--hand", "left", "--finger", "INDEX",
                 "--yes"], controller=ctl)
    assert r.code == cli.EXIT_REJECTED_MINUTIAE and r.prompts == [] and len(ctl.calls) == 1


def test_flags_without_yes_still_confirm(images):
    ctl = FakeController([OK])
    r = run_cli(["enroll", "--image", str(images.a), "--user", "5", "--hand", "Left", "--finger", "index"],
                ["y"], ctl)
    assert r.code == 0 and r.prompts == ["Proceed? [y/N]: "] and len(ctl.calls) == 1


class ScriptedQualityStages(FakeStages):
    """Raw NFIQ 2 scores taken in order, one per presentation."""

    def __init__(self, qualities):
        super().__init__()
        self.qualities = list(qualities)

    def assess_quality(self, image):
        self.calls.append("quality_before")
        return QualityResult(score=self.qualities.pop(0), actionable_feedback={}, raw_output="",
                             tool_version="test", duration_ms=0.0)


def test_session_runs_are_persisted_with_the_real_controller(tmp_path, images):
    repo = Repository(tmp_path / "db" / "fpv.sqlite", tmp_path / "templates", tmp_path / "probes",
                      artifact_root=tmp_path)
    repo.migrate()
    catalog = Catalog({}, {}, tmp_path / "raw", {})

    def factory(conf):
        return contextlib.nullcontext(PipelineController(conf, repo, catalog, ScriptedQualityStages([1, 2, 40]),
                                                         pipeline_version="0.0.0+gtest"))

    r = run_cli(["enroll"], [str(images.a), *CLAIM, "y", str(images.b), str(images.c)], factory)
    assert r.code == 0 and r.out[-1] == "Enrollment successful (template v1)"
    rows = repo.connection.execute(
        "SELECT session_id, presentation_no, status FROM runs ORDER BY presentation_no").fetchall()
    assert [(row["presentation_no"], row["status"]) for row in rows] == [
        (1, "REJECTED_QUALITY"), (2, "REJECTED_QUALITY"), (3, "ENROLLED")]
    assert len({row["session_id"] for row in rows}) == 1
    repo.close()

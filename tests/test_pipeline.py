"""Unit tests for the pipeline controller with fake stages and a temporary store (Task 9).

Covers every Error Handling branch of `enroll` / `verify`, the stage event trail, run metadata,
template provenance and ground-truth recording (Req 4, 5, 8, 9, 11). No toolchain needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fpv import config as C
from fpv.adapter import XytMinutia, XytTemplate, read_xyt
from fpv.catalog import Catalog
from fpv.matcher import InsufficientMinutiae, MatcherError, MatchResult
from fpv.pipeline import CachedProbe, PipelineController
from fpv.quality import QualityResult, QualityToolError
from fpv.storage import Repository
from fpv.types import Identity, Status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAPE = (103, 96)
ID1 = Identity.parse(1, "Left", "index")
ID2 = Identity.parse(2, "Left", "index")
ACQ = ["preprocess", "quality_after", "extract", "adapt"]


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    conf = C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT)
    assert (conf.thresholds.nfiq2_min, conf.thresholds.minutiae_min, conf.max_attempts) == (5, 12, 3)
    return conf


def _template(n: int) -> XytTemplate:
    return XytTemplate(tuple(XytMinutia(3 * (i % 30), 3 * (i // 30), (7 * i) % 360, 60) for i in range(n)))


class FakeStages:
    """Configurable stand-ins for every stage; records the call order."""

    def __init__(self, *, quality: int | None = 40, quality_exc: Exception | None = None,
                 counts: dict[int, int] | None = None, score: int = 60, match_exc: Exception | None = None):
        self.quality = quality
        self.quality_exc = quality_exc
        self.counts = counts if counts is not None else {1: 20}
        self.score = score
        self.match_exc = match_exc
        self.calls: list[str] = []

    def assess_quality(self, image):
        self.calls.append("quality_before")
        if self.quality_exc is not None:
            raise self.quality_exc
        return QualityResult(score=self.quality, actionable_feedback={"UniformImage": 1.0}, raw_output="",
                             tool_version="test", duration_ms=0.0)

    def preprocess(self, image, attempt, configuration_id):
        self.calls.append("preprocess")
        return SimpleNamespace(attempt=attempt, original_shape=SHAPE, scale_factor=attempt.scale_factor,
                               configuration_id=configuration_id, timings_ms={"enhancement": 1.0})

    def quality_after(self, pre):
        self.calls.append("quality_after")
        return QualityResult(score=50 + pre.attempt.attempt_no, actionable_feedback={}, raw_output="",
                             tool_version="test", duration_ms=0.0)

    def extract(self, pre):
        self.calls.append("extract")
        n = self.counts.get(pre.attempt.attempt_no, 0)
        return SimpleNamespace(minutiae=(), count=n, configuration_id=pre.configuration_id)

    def adapt(self, ex, pre):
        self.calls.append("adapt")
        return _template(ex.count)

    def match(self, probe, gallery):
        self.calls.append("match")
        if self.match_exc is not None:
            raise self.match_exc
        return MatchResult(self.score, read_xyt(probe).count, read_xyt(gallery).count, "test", 0.0)


@pytest.fixture
def env(tmp_path: Path, conf):
    raw = tmp_path / "raw"
    img = (np.indices(SHAPE).sum(axis=0) % 7 * 30).astype(np.uint8)
    paths = {
        "real1": raw / "Real" / "1__M_Left_index_finger.BMP",
        "cr1": raw / "Altered" / "Altered-Easy" / "1__M_Left_index_finger_CR.BMP",
        "real2": raw / "Real" / "2__F_Left_index_finger.BMP",
        "unlabeled": tmp_path / "elsewhere" / "probe.bmp",
        "broken": raw / "Real" / "3__M_Left_index_finger.BMP",
    }
    for key, p in paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if key == "broken":
            p.write_bytes(b"not an image")
        else:
            assert cv2.imwrite(str(p), img)
    repo = Repository(tmp_path / "db" / "fpv.sqlite", tmp_path / "templates", tmp_path / "probes",
                      artifact_root=tmp_path)
    repo.migrate()
    catalog = Catalog({}, {}, raw, {})

    def controller(stages: FakeStages, **kw) -> PipelineController:
        return PipelineController(conf, repo, catalog, stages, pipeline_version="0.0.0+gtest", **kw)

    yield SimpleNamespace(paths=paths, repo=repo, controller=controller)
    repo.close()


def trail(repo: Repository, run_id: str) -> list[tuple[str, int | None, str]]:
    events = repo.stage_events(run_id)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    for e in events:
        assert e["duration_ms"] >= 0 and e["started_at"].endswith("+00:00")
    return [(e["stage"], e["attempt_no"], e["outcome"]) for e in events]


def assert_run(repo: Repository, run_id: str, status: Status, reason: str | None, attempts: int) -> dict:
    run = repo.get_run(run_id)
    assert (run["status"], run["status_reason"], run["attempts_used"]) == (status.value, reason, attempts)
    assert run["ended_at"] is not None and run["ended_at"] >= run["started_at"]
    return run


def enroll(env, stages=None, path="real1"):
    return env.controller(stages or FakeStages()).enroll(env.paths[path], ID1, session_id="sess")


# --------------------------------------------------------------------------- acquisition branches


@pytest.mark.parametrize("quality, reason", [(3, "below_threshold"), (None, "nfiq2_failed")])
def test_raw_quality_rejection(env, quality, reason):
    out = enroll(env, FakeStages(quality=quality))
    assert (out.status, out.reason, out.quality_before, out.attempts) == (Status.REJECTED_QUALITY, reason,
                                                                          quality, ())
    assert out.template_id is None
    assert_run(env.repo, out.run_id, Status.REJECTED_QUALITY, reason, 0)
    assert trail(env.repo, out.run_id) == [("load", None, "ok"), ("quality_before", None, "rejected")]
    assert env.repo.stage_events(out.run_id)[1]["details"]["reason"] == reason


def test_quality_tool_error_is_error(env):
    out = enroll(env, FakeStages(quality_exc=QualityToolError("nfiq2 missing")))
    assert (out.status, out.reason) == (Status.ERROR, "nfiq2_tool_error")
    assert_run(env.repo, out.run_id, Status.ERROR, "nfiq2_tool_error", 0)
    assert trail(env.repo, out.run_id)[-1] == ("quality_before", None, "error")
    assert env.repo.stage_events(out.run_id)[-1]["details"]["error_type"] == "QualityToolError"


def test_unreadable_image_is_rejected_quality(env):
    out = enroll(env, path="broken")
    assert (out.status, out.reason) == (Status.REJECTED_QUALITY, "unreadable_image")
    assert_run(env.repo, out.run_id, Status.REJECTED_QUALITY, "unreadable_image", 0)
    assert trail(env.repo, out.run_id) == [("load", None, "error")]


def test_retry_then_enrolled_with_provenance(env, conf):
    stages = FakeStages(counts={1: 5, 2: 20})
    out = enroll(env, stages)
    assert (out.status, out.reason, out.template_version) == (Status.ENROLLED, None, 1)
    assert out.message == "Enrollment successful (template v1)"
    assert [(a.attempt_no, a.minutiae_count) for a in out.attempts] == [(1, 5), (2, 20)]
    assert_run(env.repo, out.run_id, Status.ENROLLED, None, 2)
    assert trail(env.repo, out.run_id) == (
        [("load", None, "ok"), ("quality_before", None, "ok")]
        + [(s, 1, "rejected" if s == "adapt" else "ok") for s in ACQ]
        + [(s, 2, "ok") for s in ACQ] + [("persist", 2, "ok")])
    cids = [e["configuration_id"] for e in env.repo.stage_events(out.run_id)]
    assert cids[2:6] == [conf.configuration_id(1)] * 4 and cids[6:] == [conf.configuration_id(2)] * 5

    tpl = env.repo.get_active_template(ID1)
    assert tpl.template_id == out.template_id and tpl.minutiae_count == 20
    assert (tpl.quality_before, tpl.configuration_id, tpl.scale_factor) == (40, conf.configuration_id(2), 3.0)
    assert tpl.pipeline_version == "0.0.0+gtest"
    # quality_after is stored per executed attempt (Req 8.2).
    assert [(e.attempt_no, e.configuration_id, e.score) for e in tpl.quality_after] == [
        (1, conf.configuration_id(1), 51), (2, conf.configuration_id(2), 52)]
    raw = env.repo.connection.execute("SELECT quality_after FROM templates").fetchone()[0]
    assert [e["score"] for e in json.loads(raw)] == [51, 52]


def test_all_attempts_short_is_rejected_minutiae(env):
    out = enroll(env, FakeStages(counts={1: 3, 2: 11, 3: 0}))
    assert (out.status, out.reason, len(out.attempts)) == (Status.REJECTED_MINUTIAE, "insufficient_minutiae", 3)
    assert_run(env.repo, out.run_id, Status.REJECTED_MINUTIAE, "insufficient_minutiae", 3)
    assert [s for s, _, _ in trail(env.repo, out.run_id)] == ["load", "quality_before"] + ACQ * 3
    assert env.repo.get_active_template(ID1) is None


def test_persist_failure_keeps_old_template(env, monkeypatch):
    v1 = enroll(env)
    assert v1.status is Status.ENROLLED

    def boom(path):
        raise OSError("disk full")

    monkeypatch.setattr(env.repo, "_after_artifact_write", boom)
    out = enroll(env)
    assert (out.status, out.reason, out.template_id) == (Status.ERROR, "persist_error", None)
    assert_run(env.repo, out.run_id, Status.ERROR, "persist_error", 1)
    last = env.repo.stage_events(out.run_id)[-1]
    assert (last["stage"], last["outcome"], last["details"]["error_type"]) == ("persist", "error", "OSError")
    assert env.repo.get_active_template(ID1).template_id == v1.template_id
    assert [t.template_version for t in env.repo.list_templates(ID1)] == [1]


# --------------------------------------------------------------------------- verification branches


def verify(env, stages, path="cr1", identity=ID1):
    return env.controller(stages).verify(env.paths[path], identity, session_id="sess", presentation_no=2)


def test_not_enrolled(env):
    stages = FakeStages()
    out = verify(env, stages)
    assert (out.status, out.reason, out.message, out.score) == (Status.NOT_ENROLLED, None, "Not enrolled", None)
    assert_run(env.repo, out.run_id, Status.NOT_ENROLLED, None, 1)
    assert [s for s, _, _ in trail(env.repo, out.run_id)] == ["load", "quality_before"] + ACQ + ["persist", "lookup"]
    assert "match" not in stages.calls
    assert env.repo.probe_path(out.run_id).is_file()  # the probe is stored before the lookup (Req 9.1)
    assert env.repo.get_verification(out.run_id) is None


@pytest.mark.parametrize("delta, status, message", [(0, Status.MATCH, "Match"), (25, Status.MATCH, "Match"),
                                                    (-1, Status.NO_MATCH, "No match")])
def test_decision_against_threshold(env, conf, delta, status, message):
    threshold = conf.thresholds.bozorth3_match
    enrolled = enroll(env)
    out = verify(env, FakeStages(score=threshold + delta))
    assert (out.status, out.message, out.score, out.threshold) == (status, message, threshold + delta, threshold)
    assert out.template_id == enrolled.template_id
    assert_run(env.repo, out.run_id, status, None, 1)
    assert trail(env.repo, out.run_id) == (
        [("load", None, "ok"), ("quality_before", None, "ok")] + [(s, 1, "ok") for s in ACQ]
        + [("persist", 1, "ok"), ("lookup", None, "ok"), ("match", None, "ok"), ("decide", None, "ok")])
    row = env.repo.get_verification(out.run_id)
    assert (row["decision"], row["score"], row["threshold"], row["template_id"]) == (
        status.value, threshold + delta, threshold, enrolled.template_id)
    assert env.repo.resolve_location(row["probe_location"]) == env.repo.probe_path(out.run_id)


@pytest.mark.parametrize("exc, status, reason", [
    (InsufficientMinutiae("probe", 8), Status.REJECTED_MINUTIAE, "probe_below_bozorth_min"),
    (InsufficientMinutiae("gallery", 9), Status.ERROR, "reference_below_bozorth_min"),
    (MatcherError("bozorth3 exited 1"), Status.ERROR, "bozorth3_tool_error"),
])
def test_match_failures(env, exc, status, reason):
    enroll(env)
    out = verify(env, FakeStages(match_exc=exc))
    assert (out.status, out.reason, out.score) == (status, reason, None)
    assert_run(env.repo, out.run_id, status, reason, 1)
    last = env.repo.stage_events(out.run_id)[-1]
    assert (last["stage"], last["outcome"], last["details"]["error_type"]) == ("match", "error",
                                                                                type(exc).__name__)
    assert env.repo.get_verification(out.run_id) is None


def test_verify_uses_same_quality_gate(env):
    out = verify(env, FakeStages(quality=2))
    assert (out.status, out.reason) == (Status.REJECTED_QUALITY, "below_threshold")


def test_unexpected_exception_still_ends_run(env, monkeypatch):
    enroll(env)

    def broken(identity):
        raise RuntimeError("db gone")

    monkeypatch.setattr(env.repo, "get_active_template", broken)
    out = verify(env, FakeStages())
    assert (out.status, out.reason) == (Status.ERROR, "unexpected_error")
    assert_run(env.repo, out.run_id, Status.ERROR, "unexpected_error", 1)
    assert trail(env.repo, out.run_id)[-1] == ("lookup", None, "error")


def test_run_metadata(env, conf):
    out = enroll(env)
    run = env.repo.get_run(out.run_id)
    assert (run["control_mode"], run["orchestrator_vision"]) == ("controller", "off")
    assert (run["pipeline_version"], run["config_set_id"]) == ("0.0.0+gtest", conf.config_set_id)
    assert (run["operation"], run["session_id"], run["presentation_no"]) == ("enroll", "sess", 1)
    assert (run["claimed_user_id"], run["claimed_hand"], run["claimed_finger"]) == (1, "Left", "index")
    sample = env.repo.connection.execute("SELECT * FROM samples WHERE sample_id = ?",
                                         (run["sample_id"],)).fetchone()
    assert sample["source_path"] == "Real/1__M_Left_index_finger.BMP" and sample["subject_id"] == 1


# --------------------------------------------------------------------------- ground truth


@pytest.mark.parametrize("path, expected", [("cr1", True), ("real2", False), ("unlabeled", None)])
def test_ground_truth_genuine(env, path, expected):
    enroll(env)
    out = verify(env, FakeStages(score=55), path=path)
    assert out.status is Status.MATCH and out.ground_truth_genuine is expected
    gt = env.repo.get_verification(out.run_id)["ground_truth_genuine"]
    assert gt == (None if expected is None else int(expected))


def test_labels_do_not_change_the_decision(env):
    # Identical pixels under a genuine and an impostor label: same score, same decision.
    enroll(env)
    for score in (10, 70):
        a = verify(env, FakeStages(score=score), path="cr1")
        b = verify(env, FakeStages(score=score), path="real2")
        assert (a.ground_truth_genuine, b.ground_truth_genuine) == (True, False)
        assert (a.status, a.score) == (b.status, b.score)


# --------------------------------------------------------------------------- probe cache hook


class DictCache:
    def __init__(self):
        self.data: dict = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value


def test_probe_cache_reuses_processed_probe(env, conf):
    enroll(env)
    cache, stages = DictCache(), FakeStages()
    ctl = env.controller(stages, probe_cache=cache)
    first = ctl.verify(env.paths["cr1"], ID1, session_id="s")
    second = ctl.verify(env.paths["cr1"], ID2, session_id="s")
    assert stages.calls.count("quality_before") == 1 and stages.calls.count("preprocess") == 1
    (key, value), = cache.data.items()
    assert key[1] == conf.config_set_id and isinstance(value, CachedProbe)
    assert first.status is Status.MATCH and second.status is Status.NOT_ENROLLED
    assert second.attempts == first.attempts and second.quality_before == first.quality_before
    ev = env.repo.stage_events(second.run_id)
    assert [e["stage"] for e in ev] == ["load", "quality_before", "persist", "lookup"]
    assert ev[1]["outcome"] == "skipped" and ev[1]["details"]["cache_hit"] is True

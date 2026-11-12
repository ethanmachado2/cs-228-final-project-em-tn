"""Controller agent: lifecycle, run correlation and end-to-end parity with phase 1 (Task 4, Req 4, 5.6)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("a2a")

import httpx  # noqa: E402

from fpv import config as C  # noqa: E402
from fpv.agents import serve  # noqa: E402
from fpv.agents.client import A2AClient, A2AStages, result_payload  # noqa: E402
from fpv.agents.controller import controller_handlers, sample_payload  # noqa: E402
from fpv.catalog import Catalog  # noqa: E402
from fpv.storage import Repository  # noqa: E402
from fpv.telemetry import TelemetryWriter  # noqa: E402
from test_pipeline import FakeStages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL = "http://controller:8080/"
IMG = (np.indices((103, 96)).sum(axis=0) % 7 * 30).astype(np.uint8)


@pytest.fixture(scope="module")
def conf():
    return C.load(ROOT / "configs" / "pipeline.v1.yaml", root=ROOT)


def bmp(img=IMG) -> bytes:
    ok, buf = cv2.imencode(".bmp", img)
    assert ok
    return buf.tobytes()


class Session:
    """A controller app in process plus a client, run on one event loop."""

    def __init__(self, tmp_path: Path, conf, stages, catalog=None) -> None:
        self.store, self.writer = tmp_path / "store", TelemetryWriter(tmp_path / "tel", "controller")
        handlers = controller_handlers(conf, writer=self.writer, stages=stages,
                                       catalog=catalog or Catalog({}, {}, tmp_path, {}), store=self.store,
                                       uploads=tmp_path / "uploads")
        app = serve.build_app("controller", handlers, writer=self.writer, url=URL)
        self.client = A2AClient(URL, "controller", agent="harness", writer=TelemetryWriter(tmp_path / "tel", "harness"),
                                transport=httpx.ASGITransport(app=app))
        self.loop = asyncio.new_event_loop()

    def send(self, skill, data, user=1, hand="Left", finger="index", task=None, **meta):
        t = self.loop.run_until_complete(self.client.call(
            sample_payload(data, user, hand, finger), {"skill": skill, **meta},
            task_id=task.id if task else None, context_id=task.context_id if task else None))
        state, p = result_payload(t)
        return t, state, p.data

    def close(self):
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()


def test_enroll_then_verify_completed(tmp_path, conf):
    s = Session(tmp_path, conf, FakeStages(score=60))
    try:
        _, state, d = s.send("enroll", bmp())
        assert state == "TASK_STATE_COMPLETED", d
        assert d["status"] == "ENROLLED" and d["template_version"] == 1
        _, state, d = s.send("verify", bmp())
        assert state == "TASK_STATE_COMPLETED" and d["status"] == "MATCH" and d["score"] == 60
        assert "ground_truth_genuine" not in d and d["presentation_no"] == 1
    finally:
        s.close()
    # uploads are stored by content hash, so the catalog sees no label
    uploads = list((tmp_path / "uploads").iterdir())
    assert len(uploads) == 1 and uploads[0].stem == __import__("hashlib").sha256(bmp()).hexdigest()
    with Repository(s.store / "db/fpv.sqlite", s.store / "t", s.store / "p") as repo:
        runs = repo._conn.execute("SELECT control_mode, status FROM runs ORDER BY started_at").fetchall()
    assert [tuple(r) for r in runs] == [("controller", "ENROLLED"), ("controller", "MATCH")]


def test_rejection_continuation_and_presentation_cap(tmp_path, conf):
    stages = FakeStages(quality=1)  # below nfiq2_min: REJECTED_QUALITY every time
    s = Session(tmp_path, conf, stages)
    try:
        task, state, d = s.send("enroll", bmp())
        assert state == "TASK_STATE_INPUT_REQUIRED" and d["status"] == "REJECTED_QUALITY" and d["presentation_no"] == 1
        states = []
        for _ in range(conf.max_presentations - 1):
            t2, state, d = s.send("enroll", bmp(), task=task)
            assert t2.id == task.id and t2.context_id == task.context_id
            states.append((state, d["presentation_no"]))
        n = conf.max_presentations
        assert states[-1] == ("TASK_STATE_REJECTED", n) and d["task_reason"] == "presentation_limit"
        assert all(st == "TASK_STATE_INPUT_REQUIRED" for st, _ in states[:-1])
        # a new task starts a new session
        stages.quality = 40
        _, state, d = s.send("enroll", bmp())
        assert state == "TASK_STATE_COMPLETED" and d["presentation_no"] == 1
    finally:
        s.close()
    hops = [json.loads(x) for x in s.writer.path.read_text().splitlines()]
    posts = [h for h in hops if h["kind"] == "hop" and h["http_method"] == "POST"]
    assert [h["final_state"] for h in posts][:n] == ["TASK_STATE_INPUT_REQUIRED"] * (n - 1) + ["TASK_STATE_REJECTED"]


@pytest.mark.parametrize("payload_kw", [dict(user=0), dict(hand="Middle"), dict(finger="pinky")])
def test_invalid_claim_is_rejected(tmp_path, conf, payload_kw):
    stages = FakeStages()
    s = Session(tmp_path, conf, stages)
    try:
        _, state, d = s.send("verify", bmp(), **payload_kw)
    finally:
        s.close()
    assert state == "TASK_STATE_REJECTED" and d["error_code"] == "invalid_input" and stages.calls == []


def test_unreadable_upload_is_a_pipeline_rejection_as_in_phase1(tmp_path, conf):
    s = Session(tmp_path, conf, FakeStages())
    try:
        _, state, d = s.send("verify", b"not an image")
    finally:
        s.close()
    assert state == "TASK_STATE_INPUT_REQUIRED" and (d["status"], d["reason"]) == ("REJECTED_QUALITY", "unreadable_image")


# --------------------------------------------------------------------------- end to end


def _stage_apps(conf, tel: Path) -> dict:
    return {n: httpx.ASGITransport(app=serve.build_app(n, serve.stage_handlers(n, conf), writer=TelemetryWriter(tel, n)))
            for n in A2AStages.STAGE_AGENTS}


def _summary(d: dict) -> tuple:
    return (d["status"], d["reason"], d["quality_before"], d.get("score"),
            tuple((a["attempt_no"], a["configuration_id"], a["raw_count"], a["minutiae_count"]) for a in d["attempts"]))


def _local_summary(out) -> tuple:
    return (out.status.value, out.reason, out.quality_before, getattr(out, "score", None),
            tuple((a.attempt_no, a.configuration_id, a.raw_count, a.minutiae_count) for a in out.attempts))


@pytest.mark.toolchain
@pytest.mark.dataset
def test_end_to_end_parity_with_phase1_on_seeded_train_trials(tmp_path):
    from fpv.evaluate import generate_trials
    from fpv.pipeline import PipelineController

    conf = C.load()
    raw = conf.path("raw")
    catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), raw)
    plan = generate_trials(catalog, "train", seed=42, k=1, subjects=2)
    trials = [t for t in plan.trials if t.genuine][:4] + [t for t in plan.trials if not t.genuine][:4]
    needed = {t.claim for t in trials}
    enrolls = [e for e in plan.enrollments if e.identity in needed]

    tel = tmp_path / "tel"
    a2a = A2AStages(conf, writer=TelemetryWriter(tel, "controller"), transports=_stage_apps(conf, tel))
    s = Session(tmp_path, conf, a2a, catalog=catalog)
    local_repo = Repository(tmp_path / "local/db.sqlite", tmp_path / "local/t", tmp_path / "local/p",
                            artifact_root=tmp_path / "local")
    local_repo.migrate()
    local = PipelineController(conf, local_repo, catalog)
    controller_tasks = {}
    try:
        for item in [*enrolls, *trials]:
            is_enroll = item in enrolls
            path = raw / (item.sample if is_enroll else item.probe)
            ident = item.identity if is_enroll else item.claim
            lo = (local.enroll if is_enroll else local.verify)(path, ident)
            task, _, d = s.send("enroll" if is_enroll else "verify", path.read_bytes(), ident.user_id,
                                ident.hand.value, ident.finger.value,
                                trial_id=None if is_enroll else item.trial_id)
            assert _summary(d) == _local_summary(lo), item
            controller_tasks[task.id] = d["run_id"]
    finally:
        s.close()
        a2a.close()
        local_repo.close()
    # every stage hop of a run carries its controller task ID (Req 5.6)
    for name in A2AStages.STAGE_AGENTS:
        hops = [json.loads(x) for x in (tel / f"{name}.jsonl").read_text().splitlines()]
        posts = [h for h in hops if h["http_method"] == "POST"]
        assert posts and all(h["parent_task_id"] in controller_tasks for h in posts), name

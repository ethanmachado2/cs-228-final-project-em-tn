"""Feature reducer: grouping, tier masks, warm-up exclusion, determinism (Task 7, Req 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpv import telemetry_features as F


def _task(state: str, data: dict, raw: list | None = None, artifact: bool = True) -> dict:
    parts = [{"data": data}] + (raw or [])
    if artifact:
        return {"result": {"task": {"id": "x", "status": {"state": state}, "artifacts": [{"parts": parts}]}}}
    return {"result": {"task": {"id": "x", "status": {"state": state, "message": {"parts": parts}}}}}


def hop(agent: str, skill: str, t0: int, t1: int, *, task="T1", parent=None, warmup=False, state="TASK_STATE_COMPLETED",
        data=None, raw=None, req_raw=None, trial="tr1", artifact=True) -> dict:
    return {"kind": "hop", "side": "server", "rpc_method": "SendMessage", "agent": agent, "skill": skill,
            "t_start_ns": t0, "t_end_ns": t1, "latency_ms": (t1 - t0) / 1e6, "req_bytes": 100 + t0 % 7,
            "resp_bytes": 50, "task_id": task if parent is None else f"s{t0}", "parent_task_id": parent,
            "trial_id": trial, "control_mode": "controller", "warmup": warmup, "final_state": state,
            "t_wall": "2026-11-11T00:00:00Z",
            "req_body": {"params": {"message": {"parts": [{"data": {}}] + (req_raw or [])}}},
            "resp_body": _task(state, data or {}, raw, artifact)}


def write(tmp: Path, recs: dict[str, list[dict]]) -> Path:
    for role, rs in recs.items():
        (tmp / f"{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (tmp / "blobs").mkdir(exist_ok=True)
    return tmp


@pytest.fixture
def telemetry(tmp_path):
    ms = 1_000_000
    return write(tmp_path, {
        "controller": [
            hop("controller", "verify", 0, 100 * ms, data={"run_id": "R1", "status": "MATCH", "score": 296,
                                                           "presentation_no": 1},
                req_raw=[{"raw": {"$blob": "nope", "len": 9750}, "filename": ""}]),
            hop("controller", "verify", 200 * ms, 300 * ms, task="T2", warmup=True, data={"run_id": "R2"}),
            hop("controller", "verify", 400 * ms, 450 * ms, task="T1", state="TASK_STATE_INPUT_REQUIRED",
                data={"run_id": "R3", "status": "REJECTED_QUALITY", "presentation_no": 2}, artifact=False),
        ],
        "quality": [hop("quality", "assess_quality", 1 * ms, 10 * ms, parent="T1", data={"score": 54}),
                    hop("quality", "quality_after", 30 * ms, 40 * ms, parent="T1", data={"score": 57}),
                    hop("quality", "assess_quality", 201 * ms, 210 * ms, parent="T2", warmup=True, data={"score": 1}),
                    hop("quality", "assess_quality", 401 * ms, 410 * ms, parent="T1", data={"score": 2})],
        "preprocess": [hop("preprocess", "preprocess", 11 * ms, 29 * ms, parent="T1", data={},
                           raw=[{"raw": {"$blob": "e", "len": 48000}, "filename": "enhanced"},
                                {"raw": {"$blob": "m", "len": 900}, "filename": "mask"}])],
        "extract": [hop("extract", "extract", 41 * ms, 50 * ms, parent="T1", data={"count": 35})],
        "adapter": [hop("adapter", "adapt", 51 * ms, 52 * ms, parent="T1", data={"template_count": 35})],
        "matcher": [hop("matcher", "match", 60 * ms, 90 * ms, parent="T1", data={"score": 296})],
    })


def test_runs_group_their_stage_hops_and_skip_warmup(telemetry):
    rows = F.reduce(telemetry)
    assert [r["run_id"] for r in rows] == ["R1", "R3"]  # R2 is warm-up
    r1, r3 = rows
    assert r1["t2__hops"] == 6 and r1["t2__n_preprocess"] == 1 and r1["t2__bigram_preprocess__quality_after"] == 1
    assert (r1["t3__quality_before"], r1["t3__quality_after_1"], r1["t3__raw_minutiae_1"], r1["t3__minutiae_1"]) == \
        (54, 57, 35, 35)
    assert r1["t3__enhanced_png_bytes_1"] == 48000 and r1["t3__quality_after_2"] is None
    assert (r1["oracle__score"], r1["oracle__match"], r1["cov__input_bytes"]) == (296, 1, 9750)
    assert r1["t2__ctrl_completed"] == 1 and r1["t1__run_latency_ms"] == 100.0
    # the continued presentation of task T1 only gets the hops inside its own time window
    assert r3["t2__hops"] == 1 and r3["t3__quality_before"] == 2 and r3["t2__ctrl_input_required"] == 1
    assert r3["oracle__match"] == 0 and r3["t2__presentation_no"] == 2


def test_reduce_is_deterministic(telemetry):
    assert F.reduce(telemetry) == F.reduce(telemetry)


def test_tier_masks_partition_the_feature_columns(telemetry):
    cols = list(F.reduce(telemetry)[0])
    groups = {t: set(F.tier_columns(cols, t)) for t in F.TIERS}
    feature_cols = {c for c in cols if "__" in c and not c.startswith("cov__")} - {"oracle__status"}
    assert set().union(*groups.values()) == feature_cols
    assert sum(len(g) for g in groups.values()) == len(feature_cols)  # disjoint
    assert not any(c.startswith(("oracle__", "cov__")) for t in ("t1", "t2", "t3") for c in groups[t])
    assert {"oracle__score", "oracle__match"} == groups["oracle"]


def test_labels_join_last_and_drop_warmup_and_enrollments(telemetry, tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(x) for x in (
        {"trial_id": "tr1", "arm": "C", "genuine": True, "alteration": "CR", "difficulty": "Easy", "subject": 1,
         "split": "train", "warmup": False},
        {"trial_id": "e1", "arm": "enroll", "warmup": True})) + "\n")
    rows = F.join_labels(F.reduce(telemetry), F.read_labels(labels))
    assert len(rows) == 2 and all(r["arm"] == "C" and r["label_warmup"] is False for r in rows)
    assert list(rows[0])[-len(F.LABEL_COLUMNS):] == list(F.LABEL_COLUMNS)


def test_cv_auc_separates_and_does_not_on_noise():
    pytest.importorskip("sklearn")
    import numpy as np

    rng = np.random.default_rng(0)
    rows = [{"subject": i // 2, "t1__x": float(i % 2) + rng.normal(0, 0.1), "t1__noise": rng.normal()}
            for i in range(80)]
    y = [i % 2 for i in range(80)]
    assert F.cv_auc(rows, y, ["t1__x"]) > 0.95
    assert abs(F.cv_auc(rows, y, ["t1__noise"]) - 0.5) < 0.25

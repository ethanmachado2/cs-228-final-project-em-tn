"""Arm definitions, determinism and label isolation of the A2A harness (Task 6, Req 7)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from fpv import arms
from fpv import config as C
from test_evaluate import EXCLUDED_REAL, make_catalog, synthetic_keys

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def catalog(tmp_path):
    return make_catalog(tmp_path / "raw")


def test_arm_definitions(catalog):
    plan = arms.build_arm_plan(catalog, "train", seed=42, k=2, per_arm=1000)
    by = Counter(t.arm for t in plan.trials)
    real = [k for k in synthetic_keys() if k.startswith("Real/") and k != EXCLUDED_REAL]
    assert by["A"] == len(real)
    for t in plan.trials:
        if t.arm == "A":
            assert t.probe.startswith("Real/") and t.claim == t.probe_identity and t.genuine
        elif t.arm == "B":
            assert t.probe.startswith("Real/") and not t.genuine
            assert t.claim.user_id != t.probe_identity.user_id
            assert (t.claim.hand, t.claim.finger) == (t.probe_identity.hand, t.probe_identity.finger)
        elif t.arm == "C":
            assert t.probe.startswith("Altered/") and t.genuine and t.claim == t.probe_identity and t.alteration
        else:
            assert t.probe.startswith("Altered/") and not t.genuine and t.claim != t.probe_identity
    assert {e.identity for e in plan.enrollments} == {t.claim for t in plan.trials}


def test_same_seed_same_plan_and_order(catalog):
    a = arms.build_arm_plan(catalog, "train", seed=7, k=2, per_arm=5)
    b = arms.build_arm_plan(catalog, "train", seed=7, k=2, per_arm=5)
    c = arms.build_arm_plan(catalog, "train", seed=8, k=2, per_arm=5)
    assert a == b and a.trials != c.trials
    assert all(n <= 5 for n in Counter(t.arm for t in a.trials).values())
    # interleaved, not grouped by arm
    assert [t.arm for t in a.trials] != sorted((t.arm for t in a.trials), key=arms.ARMS.index)


class FakeClient:
    """Stands in for the controller's A2AClient; records what goes on the wire."""

    def __init__(self) -> None:
        self.sent: list[tuple[dict, dict]] = []

    async def call(self, payload, metadata, **kw):
        from a2a.types import Artifact, Task, TaskState, TaskStatus

        from fpv.agents import codec

        self.sent.append((payload.data, metadata))
        status = "ENROLLED" if metadata["skill"] == "enroll" else "MATCH"
        return Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                    artifacts=[Artifact(artifact_id="a", parts=codec.to_parts(codec.Payload({"status": status})))])

    async def aclose(self):
        pass


def test_labels_are_written_and_never_sent(tmp_path, catalog):
    pytest.importorskip("a2a")
    raw = tmp_path / "raw"
    for key in synthetic_keys():
        (raw / key).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw / key), np.zeros((10, 10), np.uint8))
    conf = C.load(ROOT / "configs" / "pipeline.v1.yaml", root=ROOT)
    conf = replace(conf, paths={**conf.paths, "raw": str(raw)})
    plan = arms.build_arm_plan(catalog, "train", seed=1, k=1, per_arm=3)
    client = FakeClient()
    labels = tmp_path / "labels" / "labels.jsonl"
    counts = arms.run_a2a(conf, plan, labels_path=labels, warmup=2, client=client, out=lambda s: None)
    recs = [json.loads(x) for x in labels.read_text().splitlines()]
    trials = [r for r in recs if r["arm"] != "enroll"]
    assert [r["arm"] for r in trials] == [t.arm for t in plan.trials]
    assert [r["warmup"] for r in trials] == [True, True] + [False] * (len(trials) - 2)
    assert sum(counts.values()) == len(plan.enrollments) + len(plan.trials)
    # the wire carries the claim and an opaque trial id only
    for data, meta in client.sent:
        assert set(data) == {"user_id", "hand", "finger"}
        assert set(meta) <= {"skill", "control_mode", "trial_id", "warmup"}
    wire = json.dumps([m for _, m in client.sent])
    assert all(r["probe"] not in wire for r in trials)
    assert {m["trial_id"] for _, m in client.sent} == {r["trial_id"] for r in recs}


# --------------------------------------------------------------------------- scripted llm_routed arm (Task 9)


class FakeAdk:
    """The ADK server's /sessions and /run endpoints, scripted to end in `status`."""

    def __init__(self, status: str) -> None:
        self.status, self.calls = status, []

    def __call__(self, path: str, body: dict):
        self.calls.append((path, body))
        if path.endswith("/sessions"):
            return {"id": f"s{len(self.calls)}", "state": body.get("state")}
        usage = {"promptTokenCount": 700, "candidatesTokenCount": 20}
        if body["new_message"]["parts"][0]["text"] != "yes":
            return [{"content": {"parts": [{"functionCall": {"name": "confirm_claim", "args": {}}}]},
                     "usageMetadata": usage},
                    {"content": {"parts": [{"text": "Please confirm: user 1, Left index."}]}, "usageMetadata": usage}]
        return [{"content": {"parts": [{"functionCall": {"name": "verify", "args": {}}}]}, "usageMetadata": usage},
                {"content": {"parts": [{"functionResponse": {"name": "verify",
                                                             "response": {"status": self.status}}}]}},
                {"content": {"parts": [{"text": f"Result: {self.status}"}]}, "usageMetadata": usage}]


@pytest.mark.parametrize("status", ["MATCH", "NO_MATCH", "REJECTED_QUALITY", "NOT_ENROLLED"])
def test_scripted_conversation_reaches_each_terminal_status_and_records_cost(status):
    label = {"trial_id": "t1", "arm": "C", "claim": {"user_id": 1, "hand": "Left", "finger": "index"}}
    fake = FakeAdk(status)
    rec = arms.scripted_trial(fake, label, b"BMimage")
    assert rec["status"] == status and rec["reply"] == f"Result: {status}"
    assert (rec["model_calls"], rec["input_tokens"], rec["output_tokens"]) == (4, 2800, 80)
    assert rec["tool_calls"] == ["confirm_claim", "verify"]
    assert fake.calls[0][1] == {"state": {"trial_id": "t1"}}  # the trial id reaches the controller via state
    upload = fake.calls[1][1]["new_message"]["parts"][1]["inline_data"]
    assert upload["mime_type"] == "image/bmp"


def test_llm_trials_are_seeded_per_arm_and_skip_warmup():
    labels = [{"trial_id": f"{a}{i}", "arm": a, "warmup": i < 2} for a in ("A", "B", "C", "C'", "enroll")
              for i in range(10)]
    picked = arms.pick_llm_trials(labels, 3, seed=5)
    assert picked == arms.pick_llm_trials(labels, 3, seed=5)
    assert Counter(r["arm"] for r in picked) == {"A": 3, "B": 3, "C": 3, "C'": 3}
    assert not any(r["warmup"] for r in picked)

"""Stage agent server and telemetry capture, in process over `httpx.ASGITransport` (Task 2, Req 2, 5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("a2a")

import httpx  # noqa: E402

from fpv import config as C  # noqa: E402
from fpv.agents import codec, serve  # noqa: E402
from fpv.agents.client import A2AClient, result_payload  # noqa: E402
from fpv.quality import QualityResult, QualityToolError  # noqa: E402
from fpv.telemetry import TelemetryWriter, restore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL = "http://quality:8080/"


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(ROOT / "configs" / "pipeline.v1.yaml", root=ROOT)


class FakeQuality:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc, self.calls = exc, 0

    def assess_quality(self, image):
        self.calls += 1
        if self.exc:
            raise self.exc
        return QualityResult(score=int(image.mean()) % 100, actionable_feedback={"UniformImage": 0.25},
                             raw_output="", tool_version="fake", duration_ms=1.5)


class Recording(httpx.AsyncBaseTransport):
    """Inner transport that keeps the exact request and response bodies."""

    def __init__(self, app) -> None:
        self.inner, self.bodies = httpx.ASGITransport(app=app), []

    async def handle_async_request(self, request):
        resp = await self.inner.handle_async_request(request)
        body = await resp.aread()
        self.bodies.append((request.content, body))
        return httpx.Response(resp.status_code, headers=resp.headers, content=body)


def setup(tmp_path: Path, conf, stages=None):
    server_w, client_w = TelemetryWriter(tmp_path, "quality"), TelemetryWriter(tmp_path, "test")
    app = serve.build_app("quality", serve.stage_handlers("quality", conf, stages), writer=server_w, url=URL)
    rec = Recording(app)
    client = A2AClient(URL, "quality", agent="test", writer=client_w, transport=rec)
    client.recording = rec
    return app, client, server_w, client_w


def records(w: TelemetryWriter) -> list[dict]:
    return [json.loads(line) for line in w.path.read_text().splitlines()] if w.path.exists() else []


def call(client: A2AClient, payload: codec.Payload, **meta):
    async def go():
        try:
            return await client.call(payload, meta)
        finally:
            await client.aclose()
    return asyncio.run(go())


IMG = (np.indices((103, 96)).sum(axis=0) % 7 * 30).astype(np.uint8)


def test_agent_card(tmp_path, conf):
    app, *_ = setup(tmp_path, conf, FakeQuality())

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://q") as h:
            return (await h.get("/.well-known/agent-card.json")).json()
    card = asyncio.run(go())
    assert card["name"] == "quality"
    assert [s["id"] for s in card["skills"]] == ["assess_quality", "quality_after"]
    assert card["supportedInterfaces"] == [{"url": URL, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]
    assert card["capabilities"].get("streaming", False) is False


def test_valid_call_completes_and_is_captured_on_both_sides(tmp_path, conf):
    fake = FakeQuality()
    _, client, sw, cw = setup(tmp_path, conf, fake)
    task = call(client, codec.encode_image(IMG), skill="assess_quality", trial_id="t1")
    state, out = result_payload(task)
    assert state == "TASK_STATE_COMPLETED" and codec.decode_quality(out) == fake.assess_quality(IMG)

    server = [r for r in records(sw) if r["kind"] == "hop"]
    client_recs = records(cw)
    # one card fetch + one SendMessage on each side
    assert [r["http_method"] for r in server] == ["GET", "POST"] == [r["http_method"] for r in client_recs]
    s, c = server[1], client_recs[1]
    assert (s["side"], s["agent"], s["peer"]) == ("server", "quality", "test")
    assert (c["side"], c["agent"], c["peer"]) == ("client", "test", "quality")
    assert s["rpc_method"] == "SendMessage" and s["skill"] == "assess_quality" and s["trial_id"] == "t1"
    assert s["final_state"] == c["final_state"] == "TASK_STATE_COMPLETED"
    assert s["state_history"] == ["TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED"]
    assert s["req_parts"] == {"data": 1, "raw": 1} and s["resp_parts"] == {"data": 1}
    assert s["task_id"] == c["task_id"] == task.id and s["context_id"] == task.context_id
    assert (s["req_bytes"], s["resp_bytes"]) == (c["req_bytes"], c["resp_bytes"])
    assert c["t_start_ns"] <= s["t_start_ns"] <= s["t_end_ns"] <= c["t_end_ns"]
    # the blob store holds the PNG, and restoring gives back the exact wire body
    wire_req, wire_resp = client.recording.bodies[1]
    assert (s["req_bytes"], s["resp_bytes"]) == (len(wire_req), len(wire_resp))
    assert restore(s["req_body"], sw.blobs) == json.loads(wire_req)
    assert restore(c["resp_body"], cw.blobs) == json.loads(wire_resp)
    raw_ref = s["req_body"]["params"]["message"]["parts"][1]["raw"]
    assert (sw.blobs / raw_ref["$blob"]).read_bytes() == codec.encode_image(IMG).files["image"][0]


def test_invalid_input_is_rejected_without_running_the_stage(tmp_path, conf):
    fake = FakeQuality()
    _, client, sw, _ = setup(tmp_path, conf, fake)
    task = call(client, codec.Payload({"nope": 1}), skill="assess_quality")
    state, out = result_payload(task)
    assert state == "TASK_STATE_REJECTED" and out.data["error_code"] == "invalid_input" and fake.calls == 0
    assert records(sw)[-1]["error_code"] == "invalid_input"


def test_unknown_skill_is_rejected(tmp_path, conf):
    _, client, _, _ = setup(tmp_path, conf, FakeQuality())
    state, out = result_payload(call(client, codec.encode_image(IMG), skill="match"))
    assert state == "TASK_STATE_REJECTED" and out.data["error_code"] == "invalid_input"


def test_stage_exception_fails_with_its_code(tmp_path, conf):
    _, client, sw, _ = setup(tmp_path, conf, FakeQuality(QualityToolError("nfiq2 exited 3")))
    state, out = result_payload(call(client, codec.encode_image(IMG), skill="assess_quality"))
    assert state == "TASK_STATE_FAILED" and out.data["error_code"] == "nfiq2_tool_error"
    assert isinstance(codec.decode_error(out.data), QualityToolError)
    rec = records(sw)[-1]
    assert rec["final_state"] == "TASK_STATE_FAILED" and rec["state_history"][-1] == "TASK_STATE_FAILED"


def test_task_store_forgets_the_oldest_tasks():
    from a2a.server.context import ServerCallContext
    from a2a.types import Task

    store, ctx = serve.bounded_task_store(3), ServerCallContext()

    async def go():
        for i in range(5):
            await store.save(Task(id=f"t{i}", context_id="c"), ctx)
        return [await store.get(f"t{i}", ctx) is not None for i in range(5)]
    assert asyncio.run(go()) == [False, False, True, True, True]


def test_capture_off_changes_nothing(tmp_path, conf):
    fake = FakeQuality()
    app = serve.build_app("quality", serve.stage_handlers("quality", conf, fake), writer=None, url=URL)
    client = A2AClient(URL, "quality", agent="test", writer=None, transport=httpx.ASGITransport(app=app))
    state, out = result_payload(call(client, codec.encode_image(IMG), skill="assess_quality"))
    assert state == "TASK_STATE_COMPLETED" and codec.decode_quality(out) == fake.assess_quality(IMG)
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.toolchain
@pytest.mark.dataset
def test_quality_agent_matches_local_stages(tmp_path, conf):
    from fpv.imaging import load_gray8
    from fpv.pipeline import LocalStages

    img = load_gray8(ROOT / "data/raw/Real/1__M_Left_index_finger.BMP")
    _, client, _, _ = setup(tmp_path, conf)
    _, out = result_payload(call(client, codec.encode_image(img), skill="assess_quality"))
    remote, local = codec.decode_quality(out), LocalStages(conf).assess_quality(img)
    # raw_output differs only in NFIQ 2's temp file name
    assert (remote.score, dict(remote.actionable_feedback), remote.scored_shape) == \
        (local.score, dict(local.actionable_feedback), local.scored_shape)

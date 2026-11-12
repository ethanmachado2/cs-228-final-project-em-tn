"""Orchestrator guardrails with a fake model (Task 8, Req 9). Runs in the orchestrator image:

    docker compose run --rm --no-deps orchestrator python -m pytest -p no:cacheprovider tests/test_orchestrator.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

import fpv_orchestrator.agent as A  # noqa: E402
from fpv.telemetry import TelemetryWriter  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake-fingerprint-pixels"
REF = hashlib.sha256(PNG).hexdigest()
CLAIM = {"user_id": 1, "hand": "Left", "finger": "index"}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "UPLOADS", tmp_path / "uploads")
    writer = TelemetryWriter(tmp_path / "tel", "orchestrator")
    monkeypatch.setattr(A, "WRITER", writer)
    return writer


def events(writer: TelemetryWriter, kind: str) -> list[dict]:
    if not writer.path.exists():
        return []
    return [r for r in map(json.loads, writer.path.read_text().splitlines()) if r["kind"] == kind]


def ctx(state=None, user_text="", invocation="inv1"):
    user = types.Content(role="user", parts=[types.Part(text=user_text)])
    return SimpleNamespace(state=state if state is not None else {}, user_content=user, invocation_id=invocation)


def upload_request() -> LlmRequest:
    return LlmRequest(contents=[types.Content(role="user", parts=[
        types.Part(text="please verify me"), types.Part(inline_data=types.Blob(mime_type="image/png", data=PNG))])])


def test_uploaded_pixels_never_reach_the_model(env):
    req = upload_request()
    A.before_model(ctx(), req)
    parts = req.contents[0].parts
    assert all(p.inline_data is None for p in parts)
    assert parts[1].text == f"[sample_ref: {REF}]"
    assert PNG not in req.model_dump_json().encode() and (A.UPLOADS / REF).read_bytes() == PNG
    assert events(env, "upload")[0]["refs"] == [REF]


@pytest.mark.parametrize("bad", [{"user_id": 0}, {"user_id": 601}, {"hand": "Middle"}, {"finger": "pinky"},
                                 {"user_id": "one"}])
def test_invalid_arguments_are_blocked(env, bad):
    out = A.before_tool(SimpleNamespace(name="confirm_claim"), {**CLAIM, **bad}, ctx())
    assert out["status"] == "blocked" and "invalid claim" in out["error"]
    assert events(env, "guardrail")[-1]["type"] == "guardrail_block"


def test_unknown_sample_ref_is_blocked(env):
    state = {A.CONFIRMED: CLAIM}
    out = A.before_tool(SimpleNamespace(name="verify"), {"sample_ref": "f" * 64, **CLAIM}, SimpleNamespace(state=state))
    assert out["status"] == "blocked" and "sample_ref" in out["error"]


def test_enroll_or_verify_needs_the_users_yes(env):
    A.before_model(ctx(), upload_request())  # stores the upload
    state: dict = {}
    tool_ctx = SimpleNamespace(state=state)
    args = {"sample_ref": REF, "user_id": 1, "hand": "left", "finger": "Index"}
    assert A.before_tool(SimpleNamespace(name="verify"), dict(args), tool_ctx)["status"] == "blocked"
    A.confirm_claim(1, "Left", "index", tool_ctx)
    assert A.before_tool(SimpleNamespace(name="verify"), dict(args), tool_ctx)["status"] == "blocked"
    A.before_model(ctx(state, "no, wait"), LlmRequest())  # not a yes
    assert state[A.CONFIRMED] is None
    A.before_model(ctx(state, "Yes, that's right"), LlmRequest())
    assert state[A.CONFIRMED] == CLAIM and state[A.PENDING] is None
    call = dict(args)
    assert A.before_tool(SimpleNamespace(name="verify"), call, tool_ctx) is None
    assert (call["hand"], call["finger"]) == ("Left", "index")  # normalized to the confirmed claim
    other = {**args, "user_id": 2}
    assert A.before_tool(SimpleNamespace(name="verify"), other, tool_ctx)["status"] == "blocked"
    assert len(events(env, "guardrail")) == 3


def reply(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]),
                       usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=120,
                                                                                candidates_token_count=9))


@pytest.mark.parametrize("status, message, text, replaced", [
    ("MATCH", "Match", "Good news: the fingerprint does not match.", True),
    ("MATCH", "Match", "Result: Match. You are verified.", False),
    ("NO_MATCH", "No match", "It's a match!", True),
    ("NO_MATCH", "No match", "No match: the fingerprint was not verified.", False),
    ("ENROLLED", "Enrollment successful (template v1)", "Verification complete: match.", True),
    ("ENROLLED", "Enrollment successful (template v1)", "You are enrolled (template v1).", False),
    ("REJECTED_QUALITY", "Sample rejected: ...", "Match!", True),
    ("REJECTED_QUALITY", "Sample rejected: ...", "The sample was rejected; please upload another image.", False),
])
def test_contradicting_reply_is_replaced(env, status, message, text, replaced):
    state = {A.LAST: {"status": status, "message": message}, A.T_MODEL: 0}
    out = A.after_model(ctx(state), reply(text))
    if replaced:
        assert out.content.parts[0].text == message
        assert events(env, "guardrail")[-1]["type"] == "response_substituted"
    else:
        assert out is None
    assert state[A.LAST] is None
    call = events(env, "model_call")[-1]
    assert (call["input_tokens"], call["output_tokens"], call["model_id"]) == (120, 9, A.MODEL_ID)


def test_present_continues_an_input_required_task(env, monkeypatch):
    """`enroll` / `verify` send the upload to the controller and keep the task for the next upload."""
    from a2a.types import Artifact, Message, Task, TaskState, TaskStatus

    from fpv.agents import codec

    sent = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def call(self, payload, meta, task_id=None, context_id=None):
            sent.append((payload, meta, task_id, context_id))
            if len(sent) == 1:
                msg = Message(message_id="m", parts=codec.to_parts(codec.Payload(
                    {"status": "REJECTED_QUALITY", "message": "Sample rejected"})))
                return Task(id="T1", context_id="C1",
                            status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED, message=msg))
            return Task(id="T1", context_id="C1", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                        artifacts=[Artifact(artifact_id="a", parts=codec.to_parts(codec.Payload(
                            {"status": "MATCH", "message": "Match", "score": 296, "threshold": 43})))])

        async def aclose(self):
            pass

    import asyncio

    monkeypatch.setattr(A, "A2AClient", FakeClient)
    A.before_model(ctx(), upload_request())
    state = {A.CONFIRMED: CLAIM, "trial_id": "t1"}
    tc = SimpleNamespace(state=state)
    first = asyncio.run(A.verify(REF, 1, "Left", "index", tc))
    assert first["status"] == "REJECTED_QUALITY" and "new fingerprint" in first["next_step"]
    assert state[A.CTL_TASK] == {"skill": "verify", "task_id": "T1", "context_id": "C1"}
    second = asyncio.run(A.verify(REF, 1, "Left", "index", tc))
    assert (second["status"], second["score"]) == ("MATCH", 296) and state[A.CONFIRMED] is None
    assert [(t, c) for _, _, t, c in sent] == [(None, None), ("T1", "C1")]
    payload, meta, _, _ = sent[0]
    assert meta == {"skill": "verify", "control_mode": "llm_routed", "trial_id": "t1"}
    assert payload.files[""][0] == PNG and payload.data == CLAIM

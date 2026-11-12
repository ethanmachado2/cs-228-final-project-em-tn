"""Orchestrator: an ADK `LlmAgent` on Vertex AI Gemini in front of the controller agent (Req 9).

The model only routes: it collects the operation, a fingerprint upload and the claim, asks the
user to confirm the claim, then calls `enroll` or `verify`, which send the sample to the
controller over A2A (same client and capture as the harness, `control_mode=llm_routed`). Every
decision comes from the deterministic pipeline.

Guardrails (callbacks, not the prompt):
- `before_model_callback` moves every uploaded image out of the request into
  `<FPV_ORCHESTRATOR_UPLOADS>/<sha256>` and replaces it with `[sample_ref: <sha256>]`, so no pixels
  reach Vertex AI (Req 9.6). It also turns a pending claim into the confirmed claim when the user
  answers yes.
- `before_tool_callback` validates arguments and blocks `enroll` / `verify` unless the claim equals
  the confirmed claim (`guardrail_block`, Req 9.3, 9.4).
- `after_model_callback` replaces a final reply that contradicts the tool's status with the
  pipeline's own message (`response_substituted`, Req 9.5) and records every model call (Req 9.7).

Environment: `FPV_MODEL_ID` (default gemini-3.8-flash), `GOOGLE_GENAI_USE_ENTERPRISE` /
`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (ADC credentials),
`FPV_CONTROLLER_URL`, `FPV_ORCHESTRATOR_UPLOADS`, `FPV_TELEMETRY_DIR`.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from fpv.agents import codec
from fpv.agents.client import A2AClient, result_payload
from fpv.telemetry import writer_from_env
from fpv.types import Identity

MODEL_ID = os.environ.get("FPV_MODEL_ID", "gemini-3.8-flash")
CONTROLLER_URL = os.environ.get("FPV_CONTROLLER_URL", "http://controller:8080/")
UPLOADS = Path(os.environ.get("FPV_ORCHESTRATOR_UPLOADS", "data/orchestrator/uploads"))
WRITER = writer_from_env("orchestrator")
CONTROL_MODE = "llm_routed"
REF = re.compile(r"^[0-9a-f]{64}$")
# shortcut: English-only prefix match ("yes, but user 2" still confirms the pending claim); upgrade to a
# structured confirm button in the chat UI if the demo moves beyond adk web.
YES = re.compile(r"^\s*(y|yes|yep|yeah|correct|confirm(ed)?|that'?s (right|correct)|right)\b", re.IGNORECASE)

# state keys
PENDING, CONFIRMED, CTL_TASK, LAST = "pending_claim", "confirmed_claim", "controller_task", "last_result"
T_MODEL = "temp:model_t0"


def _event(kind: str, **fields: Any) -> None:
    if WRITER is not None:
        WRITER.event(kind, **fields)


def _claim(user_id: Any, hand: Any, finger: Any) -> dict[str, Any]:
    ident = Identity.parse(user_id, hand, finger)  # raises ValueError
    return {"user_id": ident.user_id, "hand": ident.hand.value, "finger": ident.finger.value}


# --------------------------------------------------------------------------- tools


def confirm_claim(user_id: int, hand: str, finger: str, tool_context: Any) -> dict[str, Any]:
    """Record the claimed identity and ask the user to confirm it before enroll or verify.

    Args:
        user_id: Claimed SOCOFing subject ID, 1 to 600.
        hand: Left or Right.
        finger: thumb, index, middle, ring or little.
    """
    claim = _claim(user_id, hand, finger)
    tool_context.state[PENDING] = claim
    tool_context.state[CONFIRMED] = None
    return {"status": "awaiting_confirmation",
            "ask_user": f"Please confirm: user {claim['user_id']}, {claim['hand']} {claim['finger']}. Reply yes to "
                        "continue."}


async def _present(skill: str, sample_ref: str, user_id: int, hand: str, finger: str, tool_context: Any) -> dict:
    data = (UPLOADS / sample_ref).read_bytes()
    payload = codec.Payload({"user_id": int(user_id), "hand": hand, "finger": finger}, {"": (data, codec.SAMPLE)})
    cont = tool_context.state.get(CTL_TASK) or {}
    if cont.get("skill") != skill:
        cont = {}
    client = A2AClient(CONTROLLER_URL, "controller", agent="orchestrator", writer=WRITER)
    try:
        meta = {"skill": skill, "control_mode": CONTROL_MODE}
        if tool_context.state.get("trial_id"):  # set by the scripted llm_routed harness (Task 9)
            meta["trial_id"] = tool_context.state["trial_id"]
        task = await client.call(payload, meta,
                                 task_id=cont.get("task_id"), context_id=cont.get("context_id"))
    finally:
        await client.aclose()
    state, p = result_payload(task)
    d = p.data
    if state == "TASK_STATE_INPUT_REQUIRED":  # the next upload continues this session (same task)
        tool_context.state[CTL_TASK] = {"skill": skill, "task_id": task.id, "context_id": task.context_id}
        nxt = "Ask the user to upload a new fingerprint image for the same claim."
    else:
        tool_context.state[CTL_TASK] = None
        tool_context.state[CONFIRMED] = None
        nxt = "Report the result to the user." if state == "TASK_STATE_COMPLETED" else "The session is over."
    status = d.get("status") or d.get("error_code") or state
    message = d.get("message") or d.get("error") or state
    if d.get("task_reason") == "presentation_limit":
        message += " (presentation limit reached)"
    out = {"status": status, "message": message, "next_step": nxt}
    if d.get("score") is not None:
        out.update(score=d["score"], threshold=d.get("threshold"))
    return out


async def enroll(sample_ref: str, user_id: int, hand: str, finger: str, tool_context: Any) -> dict[str, Any]:
    """Enroll the uploaded fingerprint as the confirmed claimed identity.

    Args:
        sample_ref: The 64-hex sample_ref shown in place of the uploaded image.
        user_id: Confirmed user ID.
        hand: Confirmed hand.
        finger: Confirmed finger.
    """
    return await _present("enroll", sample_ref, user_id, hand, finger, tool_context)


async def verify(sample_ref: str, user_id: int, hand: str, finger: str, tool_context: Any) -> dict[str, Any]:
    """Verify the uploaded fingerprint against the confirmed claimed identity.

    Args:
        sample_ref: The 64-hex sample_ref shown in place of the uploaded image.
        user_id: Confirmed user ID.
        hand: Confirmed hand.
        finger: Confirmed finger.
    """
    return await _present("verify", sample_ref, user_id, hand, finger, tool_context)


# --------------------------------------------------------------------------- callbacks


def _strip_images(contents: list[types.Content]) -> list[str]:
    """Replace inline image parts with sample refs, in place; returns the refs stored for the first time."""
    refs = []
    for content in contents:
        for i, part in enumerate(content.parts or []):
            blob = part.inline_data
            if blob is None or not blob.data:
                continue
            sha = hashlib.sha256(blob.data).hexdigest()
            path = UPLOADS / sha
            if not path.exists():
                UPLOADS.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob.data)
                refs.append(sha)
            content.parts[i] = types.Part(text=f"[sample_ref: {sha}]")
    return refs


def before_model(callback_context: Any, llm_request: Any) -> None:
    state = callback_context.state
    refs = _strip_images(llm_request.contents)
    user = callback_context.user_content
    text = " ".join(p.text for p in (user.parts if user else []) if p.text)
    if state.get(PENDING) and YES.match(text or ""):
        state[CONFIRMED], state[PENDING] = state[PENDING], None
    state[T_MODEL] = time.perf_counter_ns()
    if refs:
        _event("upload", n=len(refs), refs=refs)
    return None


def before_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    try:
        claim = _claim(args.get("user_id"), args.get("hand"), args.get("finger"))
    except (TypeError, ValueError) as exc:
        return _block(tool.name, args, f"invalid claim: {exc}")
    if tool.name in ("enroll", "verify"):
        ref = str(args.get("sample_ref", ""))
        if not REF.match(ref) or not (UPLOADS / ref).is_file():
            return _block(tool.name, args, "sample_ref does not match an uploaded image; ask the user to upload one")
        if tool_context.state.get(CONFIRMED) != claim:
            return _block(tool.name, args, "the claim is not confirmed; call confirm_claim and wait for the user's yes")
        args.update(claim)
    return None


def _block(tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
    _event("guardrail", type="guardrail_block", tool=tool, reason=reason,
           args={k: v for k, v in args.items() if k != "sample_ref"})
    return {"status": "blocked", "error": reason}


def after_tool(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: dict[str, Any]) -> None:
    if tool.name in ("enroll", "verify") and isinstance(tool_response, dict) and tool_response.get("status") != "blocked":
        tool_context.state[LAST] = {"status": tool_response.get("status"), "message": tool_response.get("message")}
    return None


# shortcut: English keyword regexes, so paraphrases or negations they miss slip through or get replaced;
# upgrade to a structured final answer (status field) if substitutions or misses show up in telemetry.
def contradicts(text: str, status: str) -> bool:
    """Does a final reply state an outcome other than `status`?"""
    t = text.lower()
    no_match = bool(re.search(r"\b(no match|not a match|(does|did) ?n[o']t match|no_match)\b", t))
    match = bool(re.search(r"\bmatch(ed|es)?\b", t)) and not no_match
    if status == "MATCH":
        return no_match or not match
    if status == "NO_MATCH":
        return not no_match
    if status == "ENROLLED":
        return "enroll" not in t or bool(re.search(r"\b(not enrolled|fail)", t))
    if status == "NOT_ENROLLED":
        return "not enrolled" not in t
    if status.startswith("REJECTED"):
        return not re.search(r"reject|new (fingerprint|image|sample)|another|upload", t) or match
    return not re.search(r"error|fail|could not|couldn't|unable", t) or match


def after_model(callback_context: Any, llm_response: Any) -> LlmResponse | None:
    state = callback_context.state
    content = llm_response.content
    parts = (content.parts if content else None) or []
    calls = [p.function_call.name for p in parts if p.function_call]
    usage = llm_response.usage_metadata
    t0 = state.get(T_MODEL)
    _event("model_call", model_id=MODEL_ID, invocation_id=callback_context.invocation_id,
           input_tokens=getattr(usage, "prompt_token_count", None),
           output_tokens=getattr(usage, "candidates_token_count", None),
           total_tokens=getattr(usage, "total_token_count", None),
           latency_ms=(time.perf_counter_ns() - t0) / 1e6 if t0 else None, tool_calls=calls,
           control_mode=CONTROL_MODE)
    last = state.get(LAST)
    text = "".join(p.text for p in parts if p.text)
    if calls or not last or not text or llm_response.partial:
        return None
    state[LAST] = None
    if not contradicts(text, str(last["status"])):
        return None
    _event("guardrail", type="response_substituted", status=last["status"], reply=text[:500])
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=str(last["message"]))]))


INSTRUCTION = """You are the front desk of a fingerprint enrollment and verification system.
You do not judge fingerprints yourself. The pipeline decides; you report its result.

1. Ask whether the user wants to enroll or verify, and ask them to upload a fingerprint image.
   An uploaded image appears to you only as text like [sample_ref: <64 hex characters>].
2. Ask for the claimed identity: user ID (1-600), hand (Left or Right) and finger
   (thumb, index, middle, ring or little).
3. Call confirm_claim with that identity, show the user its ask_user text, and wait for their answer.
4. After the user says yes, call enroll or verify with the most recent sample_ref and the confirmed identity.
5. Report the tool's message to the user in one or two sentences, using its exact status
   (for example "Match", "No match", "Enrollment successful"). If next_step asks for a new image,
   ask the user to upload another fingerprint for the same claim, then call the same tool again
   with the new sample_ref (no new confirmation is needed).
Never invent a result, a score or a decision."""

root_agent = Agent(
    model=MODEL_ID,
    name="fpv_orchestrator",
    description="Collects a fingerprint upload and identity claim and routes it to the fpv controller agent.",
    instruction=INSTRUCTION,
    tools=[confirm_claim, enroll, verify],
    before_model_callback=before_model,
    after_model_callback=after_model,
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
)

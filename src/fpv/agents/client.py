"""A2A clients: `A2AClient` (one peer, captured) and `A2AStages` (the phase 1 `Stages` over A2A, Req 3).

Every outbound request goes through `telemetry.capturing_transport`, so the harness, the
controller and the orchestrator all produce client-side hop records of the same shape. Calls
are blocking `SendMessage` requests with `history_length=0` (the response does not echo the
request parts back). This module only needs httpx, a2a-sdk and numpy at import time; the
orchestrator image uses it without the pipeline toolchain.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import uuid
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from fpv.agents import codec
from fpv.telemetry import TelemetryWriter, capturing_transport

if TYPE_CHECKING:
    import numpy as np

    from fpv.adapter import XytTemplate
    from fpv.config import AttemptConfig, PipelineConfig
    from fpv.extract import ExtractionResult
    from fpv.matcher import MatchResult
    from fpv.preprocess import PreprocessResult
    from fpv.quality import QualityResult

T = TypeVar("T")

# Metadata the controller attaches to every stage call of a run (Req 5.6): parent_task_id,
# trial_id, control_mode, warmup.
RUN_METADATA: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("fpv_run_metadata", default={})


@contextmanager
def run_metadata(**meta: Any) -> Iterator[None]:
    token = RUN_METADATA.set({k: v for k, v in meta.items() if v is not None})
    try:
        yield
    finally:
        RUN_METADATA.reset(token)


def state_name(task: Any) -> str:
    from a2a.types import TaskState

    return TaskState.Name(task.status.state)


def result_payload(task: Any) -> tuple[str, codec.Payload]:
    """`(TASK_STATE_..., payload)`: artifacts for completed tasks, else the status message."""
    state = state_name(task)
    if state == "TASK_STATE_COMPLETED" and task.artifacts:
        return state, codec.from_parts([p for a in task.artifacts for p in a.parts])
    return state, codec.from_parts(task.status.message.parts)


class A2AClient:
    """Client for one peer agent; `agent` is this caller's name (sent as `X-FPV-Agent`)."""

    def __init__(self, url: str, peer: str, *, agent: str, writer: TelemetryWriter | None = None,
                 timeout_s: float = 300.0, transport: Any = None) -> None:
        import httpx

        self.url, self.peer, self.agent = url, peer, agent
        self.http = httpx.AsyncClient(transport=capturing_transport(writer, agent, peer, inner=transport),
                                      timeout=timeout_s)
        self._client: Any = None

    async def _get(self) -> Any:
        if self._client is None:
            from a2a.client import ClientConfig, create_client

            self._client = await create_client(self.url, ClientConfig(streaming=False, httpx_client=self.http))
        return self._client

    async def call(self, payload: codec.Payload, metadata: dict[str, Any], *, task_id: str | None = None,
                   context_id: str | None = None) -> Any:
        """Send one message and return the final `Task`."""
        from a2a.types import Message, Role, SendMessageConfiguration, SendMessageRequest

        msg = Message(role=Role.ROLE_USER, message_id=str(uuid.uuid4()), parts=codec.to_parts(payload),
                      task_id=task_id or "", context_id=context_id or "")
        msg.metadata.update(metadata)
        req = SendMessageRequest(message=msg, configuration=SendMessageConfiguration(history_length=0))
        client = await self._get()
        task = None
        async for event in client.send_message(req):
            if event.HasField("task"):
                task = event.task
        if task is None:
            raise codec.RemoteStageError(f"{self.peer}: response carried no task")
        return task

    async def aclose(self) -> None:
        await self.http.aclose()


class _Loop:
    """A private event loop on a daemon thread, for calling async clients from sync code."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="fpv-a2a-loop", daemon=True)
        self.thread.start()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class A2AStages:
    """The phase 1 `Stages` protocol, one A2A call per method (Req 3.1).

    `urls` maps agent name to base URL (default: `serve.agent_url`); `transports` optionally
    maps agent name to an httpx transport (tests use `httpx.ASGITransport`).
    """

    STAGE_AGENTS = ("quality", "preprocess", "extract", "adapter", "matcher")

    def __init__(self, conf: PipelineConfig, *, agent: str = "controller", writer: TelemetryWriter | None = None,
                 urls: dict[str, str] | None = None, transports: dict[str, Any] | None = None) -> None:
        from fpv.agents.serve import agent_url

        self._conf = conf
        self._loop = _Loop()
        transports = transports or {}
        urls = urls or {}
        self._clients = {n: A2AClient(urls.get(n) or agent_url(n), n, agent=agent, writer=writer,
                                      transport=transports.get(n)) for n in self.STAGE_AGENTS}

    def _call(self, peer: str, skill: str, payload: codec.Payload) -> codec.Payload:
        import httpx
        from a2a.client import A2AClientError

        meta = {"skill": skill, **RUN_METADATA.get()}
        try:
            task = self._loop.run(self._clients[peer].call(payload, meta))
        except (httpx.HTTPError, A2AClientError, OSError) as exc:
            raise codec.RemoteStageError(f"{peer} unreachable: {type(exc).__name__}: {exc}") from exc
        state, out = result_payload(task)
        if state == "TASK_STATE_COMPLETED":
            return out
        raise codec.decode_error(out.data)

    # ----------------------------------------------------------------- Stages protocol

    def assess_quality(self, image: np.ndarray) -> QualityResult:
        return codec.decode_quality(self._call("quality", "assess_quality", codec.encode_image(image)))

    def preprocess(self, image: np.ndarray, attempt: AttemptConfig, configuration_id: str) -> PreprocessResult:
        p = codec.encode_image(image)
        p.data.update(codec.encode_attempt(attempt, configuration_id))
        return codec.decode_preprocess(self._call("preprocess", "preprocess", p), self._conf)

    def quality_after(self, pre: PreprocessResult) -> QualityResult:
        return codec.decode_quality(self._call("quality", "quality_after", codec.encode_preprocess(pre)))

    def extract(self, pre: PreprocessResult) -> ExtractionResult:
        return codec.decode_extraction(self._call("extract", "extract", codec.encode_preprocess(pre)))

    def adapt(self, ex: ExtractionResult, pre: PreprocessResult) -> XytTemplate:
        p = codec.encode_extraction(ex)
        p.data.update(original_shape=list(pre.original_shape), scale_factor=pre.scale_factor)
        return codec.decode_xyt(self._call("adapter", "adapt", p))

    def match(self, probe: Path, gallery: Path) -> MatchResult:
        from fpv.adapter import read_xyt

        # Read locally first, so unreadable templates raise as in phase 1 (OSError, XytParseError).
        pt, gt = read_xyt(Path(probe)), read_xyt(Path(gallery))
        p, g = codec.encode_xyt(pt, "probe"), codec.encode_xyt(gt, "gallery")
        p.data.update(g.data)
        p.files.update(g.files)
        return codec.decode_match(self._call("matcher", "match", p))

    def close(self) -> None:
        for c in self._clients.values():
            self._loop.run(c.aclose())
        self._loop.close()


def present(url: str, skill: str, data: bytes, user_id: int, hand: str, finger: str, *, agent: str = "harness",
            writer: TelemetryWriter | None = None, **meta: Any) -> tuple[str, dict[str, Any], Any]:
    """Send one presentation to the controller: `(task state, outcome data, task)`."""
    from fpv.agents.controller import sample_payload

    async def go() -> Any:
        c = A2AClient(url, "controller", agent=agent, writer=writer)
        try:
            return await c.call(sample_payload(data, user_id, hand, finger), {"skill": skill, **meta})
        finally:
            await c.aclose()
    task = asyncio.run(go())
    state, p = result_payload(task)
    return state, p.data, task


def main(argv: list[str] | None = None) -> int:
    """`python -m fpv.agents.client enroll|verify IMAGE USER HAND FINGER`: one presentation to the controller."""
    import argparse
    import json

    from fpv.agents.serve import agent_url
    from fpv.telemetry import writer_from_env

    ap = argparse.ArgumentParser(prog="python -m fpv.agents.client")
    ap.add_argument("skill", choices=["enroll", "verify"])
    ap.add_argument("image", type=Path)
    ap.add_argument("user_id", type=int)
    ap.add_argument("hand")
    ap.add_argument("finger")
    ap.add_argument("--trial-id", default=None)
    args = ap.parse_args(argv)
    meta = {"trial_id": args.trial_id} if args.trial_id else {}
    state, data, task = present(agent_url("controller"), args.skill, args.image.read_bytes(), args.user_id,
                                args.hand, args.finger, writer=writer_from_env("harness"), **meta)
    print(json.dumps({"task_id": task.id, "state": state, **data}, indent=2, default=str))
    return 0 if state == "TASK_STATE_COMPLETED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

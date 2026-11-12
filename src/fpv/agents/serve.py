"""A2A servers for the phase 1 stages (Req 2) and the controller (Req 4).

`python -m fpv.agents.serve <name> [--host H] [--port P]` runs one agent under uvicorn with a
single worker. Each request message carries its skill in `metadata.skill` (optional when the
agent has one skill) and a `Payload` (see `codec`). The executor:

- ends the task `rejected` with `invalid_input` when the payload does not decode (the stage
  never runs, Req 2.5);
- runs the unchanged phase 1 stage in a worker thread, one request at a time per agent (the
  pipeline is serial in phase 1 too, and this keeps timing free of contention);
- ends `completed` with the result as an artifact, or `failed` with an error code (Req 2.3, 2.4).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor

from fpv import __version__
from fpv.agents import codec
from fpv.telemetry import TelemetryMiddleware, TelemetryWriter, note_state, writer_from_env

if TYPE_CHECKING:
    from fpv.config import PipelineConfig

Handler = Callable[[codec.Payload, dict[str, Any]], "Outcome"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a handler returns: a final task state and its payload."""

    state: str  # "completed" | "input_required" | "rejected" | "failed"
    payload: codec.Payload
    as_artifact: bool = True  # completed results go in an artifact; others in the status message


def completed(p: codec.Payload) -> Outcome:
    return Outcome("completed", p)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    description: str
    skills: dict[str, str]  # skill id -> description


AGENTS: dict[str, AgentSpec] = {s.name: s for s in (
    AgentSpec("controller", "Deterministic fpv pipeline controller (phase 1 policy over the stage agents)",
              {"enroll": "Enroll a sample as the claimed identity's active template",
               "verify": "Verify a sample against the claimed identity's active template"}),
    AgentSpec("quality", "NFIQ 2 quality scoring",
              {"assess_quality": "NFIQ 2 score of a raw sample (PNG)",
               "quality_after": "NFIQ 2 score of a preprocessing attempt's enhanced image"}),
    AgentSpec("preprocess", "pyfing preprocessing (segmentation, orientation, frequency, enhancement)",
              {"preprocess": "Run one configured preprocessing attempt on a raw sample"}),
    AgentSpec("extract", "pyfing SBMEX minutiae extraction",
              {"extract": "Extract minutiae from an attempt's enhanced image and mask"}),
    AgentSpec("adapter", "pyfing minutiae to NIST .xyt adapter",
              {"adapt": "Map minutiae to the original sample's .xyt frame"}),
    AgentSpec("matcher", "NBIS Bozorth3 matcher",
              {"match": "Bozorth3 score of a probe .xyt against a gallery .xyt"}),
)}


PORT = 8080  # every agent listens here inside its own container; compose service names tell them apart


def agent_url(name: str) -> str:
    """Peer URL from `FPV_<NAME>_URL`, defaulting to the compose service name."""
    return os.environ.get(f"FPV_{name.upper()}_URL", f"http://{name}:{PORT}/")


def agent_card(spec: AgentSpec, url: str | None = None) -> Any:
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

    return AgentCard(
        name=spec.name, description=spec.description, version=__version__,
        default_input_modes=["application/json", codec.PNG], default_output_modes=["application/json", codec.PNG],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[AgentSkill(id=k, name=k, description=v, tags=["fpv", spec.name]) for k, v in spec.skills.items()],
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0",
                                             url=url or agent_url(spec.name))])


# --------------------------------------------------------------------------- stage handlers


def stage_handlers(name: str, conf: PipelineConfig, stages: Any = None) -> dict[str, Handler]:
    """Skill handlers that decode the payload, call the phase 1 stage and encode the result."""
    from fpv.adapter import read_xyt, to_xyt, write_xyt
    from fpv.pipeline import LocalStages

    st = stages if stages is not None else LocalStages(conf)

    def assess_quality(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        return completed(codec.encode_quality(st.assess_quality(codec.decode_image(p))))

    def quality_after(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        return completed(codec.encode_quality(st.quality_after(codec.decode_preprocess(p, conf))))

    def preprocess(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        image = codec.decode_image(p)
        attempt, cid = codec.decode_attempt(p.data, conf)
        return completed(codec.encode_preprocess(st.preprocess(image, attempt, cid)))

    def extract(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        return completed(codec.encode_extraction(st.extract(codec.decode_preprocess(p, conf))))

    def adapt(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        ex = codec.decode_extraction(p)
        try:
            shape = tuple(int(v) for v in p.data["original_shape"])
            scale = float(p.data["scale_factor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise codec.InvalidInput(f"adapt needs original_shape and scale_factor: {exc}") from exc
        # Same computation as fpv.acquire.adapt(ex, pre), which only reads these two fields.
        return completed(codec.encode_xyt(to_xyt(ex.minutiae, shape, scale)))

    def match(p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        probe, gallery = codec.decode_xyt(p, "probe"), codec.decode_xyt(p, "gallery")
        with tempfile.TemporaryDirectory(prefix="fpv-match-") as tmp:
            pp, gp = Path(tmp) / "probe.xyt", Path(tmp) / "gallery.xyt"
            write_xyt(probe, pp)
            write_xyt(gallery, gp)
            assert read_xyt(pp) == probe  # the files Bozorth3 reads are the templates sent
            return completed(codec.encode_match(st.match(pp, gp)))

    table = {"assess_quality": assess_quality, "quality_after": quality_after, "preprocess": preprocess,
             "extract": extract, "adapt": adapt, "match": match}
    return {skill: table[skill] for skill in AGENTS[name].skills}


# --------------------------------------------------------------------------- executor


class StageExecutor(AgentExecutor):
    """a2a-sdk `AgentExecutor` dispatching on `metadata.skill` to synchronous handlers."""

    def __init__(self, handlers: dict[str, Handler]) -> None:
        self.handlers = handlers
        self._lock = asyncio.Lock()

    async def execute(self, ctx: Any, queue: Any) -> None:
        from a2a.helpers.proto_helpers import new_task
        from a2a.server.tasks import TaskUpdater
        from a2a.types import TaskState
        from google.protobuf.json_format import MessageToDict

        up = TaskUpdater(queue, ctx.task_id, ctx.context_id)
        if ctx.current_task is None:
            await queue.enqueue_event(new_task(ctx.task_id, ctx.context_id, TaskState.TASK_STATE_SUBMITTED,
                                               history=[ctx.message]))
            note_state(ctx.task_id, "TASK_STATE_SUBMITTED")
        meta = MessageToDict(ctx.message.metadata) if ctx.message.HasField("metadata") else {}

        async def end(state: str, p: codec.Payload, as_artifact: bool = False) -> None:
            note_state(ctx.task_id, f"TASK_STATE_{state.upper()}")
            parts = codec.to_parts(p)
            if as_artifact:
                await up.add_artifact(parts, name=skill or "result")
                await up.complete()
                return
            msg = up.new_agent_message(parts)
            await {"completed": up.complete, "input_required": up.requires_input,
                   "rejected": up.reject, "failed": up.failed}[state](msg)

        skill = meta.get("skill") or (next(iter(self.handlers)) if len(self.handlers) == 1 else None)
        handler = self.handlers.get(skill) if skill else None
        if handler is None:
            await end("rejected", codec.Payload(codec.encode_error(codec.InvalidInput(f"unknown skill {skill!r}"))))
            return
        async with self._lock:
            await up.start_work()
            note_state(ctx.task_id, "TASK_STATE_WORKING")
            try:
                payload = codec.from_parts(ctx.message.parts)
                call_meta = {**meta, "_task_id": ctx.task_id, "_context_id": ctx.context_id}
                out = await asyncio.to_thread(handler, payload, call_meta)
            except codec.InvalidInput as exc:
                await end("rejected", codec.Payload(codec.encode_error(exc)))
                return
            except Exception as exc:  # noqa: BLE001 - every stage failure becomes a failed task
                await end("failed", codec.Payload(codec.encode_error(exc)))
                return
            await end(out.state, out.payload, as_artifact=out.state == "completed" and out.as_artifact)

    async def cancel(self, ctx: Any, queue: Any) -> None:  # pragma: no cover - stages are short and uncancelable
        from a2a.utils.errors import TaskNotCancelableError

        raise TaskNotCancelableError()


# --------------------------------------------------------------------------- warm-up


def _ridges(shape: tuple[int, int] = (103, 96), period: float = 5.0) -> Any:
    """A synthetic ridge pattern at SOCOFing size, so warm-up needs no dataset (Req 6.4)."""
    import numpy as np

    y, x = np.indices(shape)
    r = np.hypot(x - shape[1] / 2, y - shape[0] / 2)
    return (127.5 + 100 * np.sin(2 * np.pi * r / period)).astype(np.uint8)


def warmup_payloads(name: str, conf: PipelineConfig) -> list[tuple[str, codec.Payload]]:
    """One synthetic request per skill; `extract` / `quality_after` get a synthetic enhanced image."""
    import numpy as np

    from fpv.adapter import XytMinutia, XytTemplate
    from fpv.preprocess import PreprocessResult

    att = conf.attempt(1)
    cid = conf.configuration_id(att)
    img = _ridges()
    big = _ridges((int(103 * att.scale_factor), int(96 * att.scale_factor)), 5.0 * att.scale_factor)
    pre = PreprocessResult(att, big, np.full(big.shape, 255, np.uint8), att.scale_factor, img.shape, cid, {})
    tpl = XytTemplate(tuple(XytMinutia(5 * i, 4 * i + 3, (29 * i) % 360, 60) for i in range(15)))
    preprocess_req = codec.encode_image(img)
    preprocess_req.data.update(codec.encode_attempt(att, cid))
    adapt_req = codec.Payload({"minutiae": [[10, 12, 0.5, "ending", 0.8]] * 3, "count": 3, "configuration_id": cid,
                               "duration_ms": 0.0, "original_shape": list(img.shape), "scale_factor": att.scale_factor})
    match_req = codec.encode_xyt(tpl, "probe")
    g = codec.encode_xyt(tpl, "gallery")
    match_req.data.update(g.data)
    match_req.files.update(g.files)
    table = {"assess_quality": codec.encode_image(img), "quality_after": codec.encode_preprocess(pre),
             "preprocess": preprocess_req, "extract": codec.encode_preprocess(pre), "adapt": adapt_req,
             "match": match_req}
    return [(skill, table[skill]) for skill in AGENTS[name].skills if skill in table]


class Health:
    """`/healthz` is 503 until the warm-up calls finished (the compose healthcheck waits on it)."""

    def __init__(self) -> None:
        self.ready = False


def warmup_lifespan(name: str, conf: PipelineConfig, port: int, health: Health) -> Any:
    """Starlette lifespan: once the server listens, call each skill on itself with `warmup: true`."""
    from fpv.agents.client import A2AClient

    async def warm() -> None:
        import httpx

        url = f"http://127.0.0.1:{port}/"
        async with httpx.AsyncClient() as h:
            for _ in range(600):
                with contextlib.suppress(httpx.HTTPError):
                    if (await h.get(url + ".well-known/agent-card.json")).status_code == 200:
                        break
                await asyncio.sleep(0.1)
        client = A2AClient(url, name, agent=f"{name}-warmup")
        try:
            for skill, payload in warmup_payloads(name, conf):
                with contextlib.suppress(Exception):  # a failed warm-up call still loaded the stage
                    await client.call(payload, {"skill": skill, "warmup": True})
        finally:
            await client.aclose()
            health.ready = True

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> Any:
        task = asyncio.create_task(warm())
        yield
        task.cancel()

    return lifespan


# shortcut: tasks beyond the newest 512 are forgotten, so continuing or fetching an older task fails;
# upgrade to a2a's DatabaseTaskStore if sessions must outlive 512 later tasks.
def bounded_task_store(max_tasks: int = 512) -> Any:
    """`InMemoryTaskStore` that forgets the oldest tasks, so an always-on agent's memory stays flat.

    Each stored task holds its request parts (images up to ~50 KB). Only the controller ever
    continues a task (`input-required`), and only the most recent ones.
    """
    from collections import OrderedDict

    from a2a.server.tasks import InMemoryTaskStore

    class BoundedTaskStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self._order: OrderedDict[str, Any] = OrderedDict()

        async def save(self, task: Any, context: Any) -> None:
            await super().save(task, context)
            self._order[task.id] = context
            self._order.move_to_end(task.id)
            while len(self._order) > max_tasks:
                old, ctx = self._order.popitem(last=False)
                await super().delete(old, ctx)

    return BoundedTaskStore()


def build_app(name: str, handlers: dict[str, Handler], *, writer: TelemetryWriter | None = None,
              url: str | None = None, lifespan: Any = None, health: Health | None = None) -> Any:
    """Starlette app: AgentCard, JSON-RPC route, `/healthz` and the telemetry middleware."""
    from a2a.server.request_handlers import DefaultRequestHandlerV2
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    health = health or Health()
    if lifespan is None:
        health.ready = True

    async def healthz(request: Any) -> Any:
        return PlainTextResponse("ok" if health.ready else "warming up", status_code=200 if health.ready else 503)

    card = agent_card(AGENTS[name], url)
    handler = DefaultRequestHandlerV2(agent_executor=StageExecutor(handlers), task_store=bounded_task_store(),
                                      agent_card=card)
    app = Starlette(routes=[*create_agent_card_routes(card), Route("/healthz", healthz),
                            *create_jsonrpc_routes(handler, "/")], lifespan=lifespan)
    app.state.card = card
    return TelemetryMiddleware(app, writer, name)


def make_app(name: str, conf: PipelineConfig | None = None, port: int = PORT) -> Any:
    from fpv import config as cfgmod

    conf = conf or cfgmod.load(os.environ.get("FPV_CONFIG"))
    writer = writer_from_env(name)
    if name == "controller":  # no models to load; the harness flags its first trials as warm-up instead
        from fpv.agents.controller import controller_handlers

        return build_app(name, controller_handlers(conf, writer=writer), writer=writer)
    health = Health()
    return build_app(name, stage_handlers(name, conf), writer=writer, health=health,
                     lifespan=warmup_lifespan(name, conf, port, health))


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m fpv.agents.serve", description="run one fpv A2A agent")
    ap.add_argument("name", choices=sorted(AGENTS))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    app = make_app(args.name, port=args.port or PORT)
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host=args.host, port=args.port or PORT, workers=1,
                    log_level=os.environ.get("FPV_LOG_LEVEL", "warning"), access_log=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

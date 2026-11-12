"""Controller agent: `enroll` / `verify` skills running the unchanged `PipelineController` (Req 4).

Request: one raw part with no filename (the sample's original bytes) plus a data part
`{"user_id", "hand", "finger"}`. Optional message metadata: `trial_id`, `control_mode`,
`warmup`. The bytes are stored as `<uploads>/<sha256>.bmp`, so the catalog sees no SOCOFing
filename and the run has no ground truth (Req 4.2).

Task states (Req 4.3-4.5): a REJECTED_QUALITY / REJECTED_MINUTIAE run ends `input-required`
while the session (`contextId`) has presentations left, and `rejected` with
`reason=presentation_limit` at `session.max_presentations`; every other status ends
`completed`. The next presentation is a new message on the same task (`taskId` + `contextId`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fpv.agents import codec
from fpv.agents.client import A2AStages, run_metadata
from fpv.agents.serve import Handler, Outcome, completed

if TYPE_CHECKING:
    from fpv.catalog import Catalog
    from fpv.config import PipelineConfig
    from fpv.telemetry import TelemetryWriter

REJECTED = ("REJECTED_QUALITY", "REJECTED_MINUTIAE")
REASON_PRESENTATION_LIMIT = "presentation_limit"


def sample_payload(data: bytes, user_id: int, hand: str, finger: str) -> codec.Payload:
    """The request a harness or orchestrator sends to `enroll` / `verify`."""
    return codec.Payload({"user_id": user_id, "hand": hand, "finger": finger}, {"": (data, codec.SAMPLE)})


def outcome_data(out: Any, operation: str, presentation_no: int) -> dict[str, Any]:
    d = {"operation": operation, "run_id": out.run_id, "status": out.status.value, "reason": out.reason,
         "message": out.message, "quality_before": out.quality_before, "presentation_no": presentation_no,
         "template_id": out.template_id,
         "attempts": [dataclasses.asdict(a) for a in out.attempts]}
    if operation == "enroll":
        d["template_version"] = out.template_version
    else:  # ground_truth_genuine stays off the wire (Req 5.7); it is None for uploads anyway
        d.update(score=out.score, threshold=out.threshold)
    return d


def controller_handlers(conf: PipelineConfig, *, writer: TelemetryWriter | None = None, stages: Any = None,
                        catalog: Catalog | None = None, store: Path | None = None,
                        uploads: Path | None = None) -> dict[str, Handler]:
    from fpv.catalog import Catalog
    from fpv.pipeline import PipelineController
    from fpv.storage import Repository
    from fpv.types import Identity

    stages = stages if stages is not None else A2AStages(conf, writer=writer)
    catalog = catalog or Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    store = store or (Path(os.environ["FPV_STORE_DIR"]) if os.environ.get("FPV_STORE_DIR") else None)
    if store is None:
        db, templates, probes, root = conf.path("db"), conf.path("templates"), conf.path("probes"), conf.root
    else:
        db, templates, probes, root = store / "db" / "fpv.sqlite", store / "templates", store / "probes", store
    uploads = uploads or Path(os.environ.get("FPV_UPLOADS_DIR") or conf.root / "data" / "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    with Repository(db, templates, probes, artifact_root=root) as repo:
        repo.migrate()
    # shortcut: contextId -> rejected presentations, in memory only, so a controller restart resets the
    # presentation cap; upgrade to counting rejected runs per session_id in fpv.sqlite if restarts mid-session matter.
    rejected_so_far: dict[str, int] = {}
    lock = threading.Lock()

    def run(operation: str, p: codec.Payload, meta: dict[str, Any]) -> Outcome:
        if "" not in p.files:
            raise codec.InvalidInput("missing the sample (a raw part with no filename)")
        data, _media = p.files[""]
        if not data:
            raise codec.InvalidInput("the sample is empty")
        try:
            identity = Identity.parse(p.data.get("user_id"), p.data.get("hand"), p.data.get("finger"))
        except (TypeError, ValueError) as exc:
            raise codec.InvalidInput(f"invalid claim: {exc}") from exc
        path = uploads / f"{hashlib.sha256(data).hexdigest()}.bmp"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)

        context_id, task_id = meta["_context_id"], meta["_task_id"]
        with lock:
            presentation_no = rejected_so_far.get(context_id, 0) + 1
        mode = meta.get("control_mode") or "controller"
        with run_metadata(parent_task_id=task_id, trial_id=meta.get("trial_id"), control_mode=mode,
                          warmup=meta.get("warmup")), \
                Repository(db, templates, probes, artifact_root=root) as repo:
            ctl = PipelineController(conf, repo, catalog, stages, control_mode=mode)
            fn = ctl.enroll if operation == "enroll" else ctl.verify
            out = fn(path, identity, session_id=context_id, presentation_no=presentation_no)
        result = codec.Payload(outcome_data(out, operation, presentation_no))
        with lock:
            if out.status.value not in REJECTED:
                rejected_so_far.pop(context_id, None)
                return completed(result)
            if presentation_no >= conf.max_presentations:
                rejected_so_far.pop(context_id, None)
                result.data["task_reason"] = REASON_PRESENTATION_LIMIT
                return Outcome("rejected", result, as_artifact=False)
            rejected_so_far[context_id] = presentation_no
            return Outcome("input_required", result, as_artifact=False)

    return {"enroll": lambda p, m: run("enroll", p, m), "verify": lambda p, m: run("verify", p, m)}

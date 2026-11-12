"""A2A hop telemetry: JSONL writer, blob store, server ASGI middleware, client httpx transport (Req 5).

Every hop is one JSON line in `<dir>/<role>.jsonl` (one writer per file, Req 5.4). Bodies are
stored as sent, except that every A2A raw part's base64 `raw` string is replaced by
`{"$blob": <sha256>, "len": <bytes>}` and the bytes are written once to `<dir>/blobs/<sha256>`;
`restore` undoes it exactly (Req 5.3). `req_bytes` / `resp_bytes` are HTTP body lengths as
sent and received. Capture is passive: write errors go to stderr and are never raised, and
`FPV_TELEMETRY=off` disables it (Req 5.5). Keep this module free of heavy imports: the
orchestrator image uses it without the pipeline toolchain.

Environment: `FPV_TELEMETRY_DIR` (default `data/telemetry`), `FPV_TELEMETRY` (`off` disables),
`FPV_IMAGE_DIGEST` (recorded on every hop).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
AGENT_HEADER = "x-fpv-agent"

# Task states an executor emitted, by task ID; the middleware pops them into the hop record.
_STATES: dict[str, list[str]] = {}
_STATES_LOCK = threading.Lock()


def note_state(task_id: str, state: str) -> None:
    with _STATES_LOCK:
        _STATES.setdefault(task_id, []).append(state)


def _pop_states(task_id: str | None) -> list[str] | None:
    if not task_id:
        return None
    with _STATES_LOCK:
        return _STATES.pop(task_id, None)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TelemetryWriter:
    """Append-only JSONL for one role plus the shared, content-addressed blob store."""

    def __init__(self, directory: Path | str, role: str) -> None:
        self.dir = Path(directory)
        self.role = role
        self.path = self.dir / f"{role}.jsonl"
        self.blobs = self.dir / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.image_digest = os.environ.get("FPV_IMAGE_DIGEST")

    def blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        path = self.blobs / sha
        if not path.exists():
            tmp = path.with_name(f".{sha}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic; identical content if another process won the race
        return sha

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        """A non-hop record (model calls, guardrail events, warm-up markers)."""
        self.safe(lambda: self.write({"schema_version": SCHEMA_VERSION, "kind": kind, "role": self.role,
                                      "t_wall": utc_now(), "t_mono_ns": time.monotonic_ns(), **fields}))

    def safe(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - capture must never affect the pipeline (Req 5.5)
            print(f"telemetry: {type(exc).__name__}: {exc}", file=sys.stderr)


def writer_from_env(role: str) -> TelemetryWriter | None:
    if os.environ.get("FPV_TELEMETRY", "on").lower() in ("off", "0", "false"):
        return None
    return TelemetryWriter(os.environ.get("FPV_TELEMETRY_DIR", "data/telemetry"), role)


# --------------------------------------------------------------------------- bodies


def scrub(body: Any, blob: Callable[[bytes], str]) -> Any:
    """Copy of a JSON body with every raw part's base64 bytes moved to the blob store."""
    if isinstance(body, dict):
        out = {k: scrub(v, blob) for k, v in body.items()}
        raw = body.get("raw")
        if isinstance(raw, str):
            try:
                data = base64.b64decode(raw, validate=True)
            except (binascii.Error, ValueError):
                return out
            if base64.b64encode(data).decode("ascii") == raw:  # only if restore is exact
                out["raw"] = {"$blob": blob(data), "len": len(data)}
        return out
    if isinstance(body, list):
        return [scrub(v, blob) for v in body]
    return body


def restore(body: Any, blobs: Path | str) -> Any:
    """Inverse of `scrub`."""
    if isinstance(body, dict):
        if set(body) == {"$blob", "len"}:
            return base64.b64encode((Path(blobs) / body["$blob"]).read_bytes()).decode("ascii")
        return {k: restore(v, blobs) for k, v in body.items()}
    if isinstance(body, list):
        return [restore(v, blobs) for v in body]
    return body


def _parts_summary(parts: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in parts or ():
        if isinstance(p, dict):
            kind = next((k for k in ("raw", "data", "text", "url") if k in p), "other")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def summarize(req: Any, resp: Any) -> dict[str, Any]:
    """Protocol fields of one JSON-RPC exchange; missing pieces are None."""
    req = req if isinstance(req, dict) else {}
    resp = resp if isinstance(resp, dict) else {}
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    msg = params.get("message") if isinstance(params.get("message"), dict) else {}
    meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
    result = resp.get("result") if isinstance(resp.get("result"), dict) else {}
    task = result.get("task") if isinstance(result.get("task"), dict) else {}
    status = task.get("status") if isinstance(task.get("status"), dict) else {}
    status_parts = (status.get("message") or {}).get("parts") or []
    error_code = next((p["data"].get("error_code") for p in status_parts
                       if isinstance(p, dict) and isinstance(p.get("data"), dict) and "error_code" in p["data"]), None)
    resp_parts: list[Any] = [p for a in task.get("artifacts") or [] for p in (a.get("parts") or [])]
    resp_parts += status_parts
    rpc_error = resp.get("error") if isinstance(resp.get("error"), dict) else None
    return {
        "rpc_method": req.get("method"),
        "rpc_error": rpc_error.get("code") if rpc_error else None,
        "skill": meta.get("skill"),
        "task_id": task.get("id") or msg.get("taskId"),
        "context_id": task.get("contextId") or msg.get("contextId"),
        "parent_task_id": meta.get("parent_task_id"),
        "trial_id": meta.get("trial_id"),
        "control_mode": meta.get("control_mode"),
        "warmup": bool(meta.get("warmup", False)),
        "final_state": status.get("state"),
        "error_code": error_code,
        "req_parts": _parts_summary(msg.get("parts")),
        "resp_parts": _parts_summary(resp_parts),
    }


def _json(data: bytes) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (UnicodeDecodeError, ValueError):
        return None


def hop_record(writer: TelemetryWriter, *, side: str, agent: str, peer: str | None, method: str, path: str,
               status: int | None, t_wall: str, t0: int, t1: int, req: bytes, resp: bytes,
               states: list[str] | None = None) -> dict[str, Any]:
    req_json, resp_json = _json(req), _json(resp)
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": "hop", "side": side, "role": writer.role, "agent": agent,
        "peer": peer, "t_wall": t_wall, "t_start_ns": t0, "t_end_ns": t1, "latency_ms": (t1 - t0) / 1e6,
        "http_method": method, "path": path, "http_status": status, "image_digest": writer.image_digest,
        "req_bytes": len(req), "resp_bytes": len(resp), **summarize(req_json, resp_json),
        "req_body": scrub(req_json, writer.blob) if req_json is not None else None,
        "resp_body": scrub(resp_json, writer.blob) if resp_json is not None else None,
    }
    if side == "server":
        rec["state_history"] = states if states is not None else _pop_states(rec["task_id"])
    return rec


# --------------------------------------------------------------------------- server side


class TelemetryMiddleware:
    """Pure ASGI middleware recording every HTTP request the agent serves (Req 5.1)."""

    def __init__(self, app: Any, writer: TelemetryWriter | None, agent: str) -> None:
        self.app, self.writer, self.agent = app, writer, agent

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or self.writer is None or scope.get("path") == "/healthz":
            await self.app(scope, receive, send)
            return
        t_wall, t0 = utc_now(), time.monotonic_ns()
        req_chunks: list[bytes] = []
        resp_chunks: list[bytes] = []
        status: list[int] = []

        async def recv() -> dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                req_chunks.append(message.get("body", b""))
            return message

        async def snd(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status.append(message["status"])
            elif message["type"] == "http.response.body":
                resp_chunks.append(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, recv, snd)
        finally:
            t1 = time.monotonic_ns()
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            writer = self.writer
            writer.safe(lambda: writer.write(hop_record(
                writer, side="server", agent=self.agent, peer=headers.get(AGENT_HEADER), method=scope["method"],
                path=scope["path"], status=status[0] if status else None, t_wall=t_wall, t0=t0, t1=t1,
                req=b"".join(req_chunks), resp=b"".join(resp_chunks))))


# --------------------------------------------------------------------------- client side


def capturing_transport(writer: TelemetryWriter | None, agent: str, peer: str, inner: Any = None) -> Any:
    """httpx transport that records every request this client sends to `peer` (Req 5.1)."""
    import httpx

    inner = inner or httpx.AsyncHTTPTransport()

    class CapturingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            request.headers[AGENT_HEADER] = agent
            req = await request.aread()
            t_wall, t0 = utc_now(), time.monotonic_ns()
            resp = await inner.handle_async_request(request)
            body = await resp.aread()
            t1 = time.monotonic_ns()
            if writer is not None:
                writer.safe(lambda: writer.write(hop_record(
                    writer, side="client", agent=agent, peer=peer, method=request.method, path=request.url.path,
                    status=resp.status_code, t_wall=t_wall, t0=t0, t1=t1, req=req, resp=body)))
            return httpx.Response(resp.status_code, headers=resp.headers, content=body,
                                  extensions=resp.extensions, request=request)

        async def aclose(self) -> None:
            await inner.aclose()

    return CapturingTransport()

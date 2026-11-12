"""Phase 1 contracts <-> A2A parts (Req 1.1-1.3).

A message or artifact is a `Payload`: one data part (a JSON object) plus named raw parts
(`Part.filename` is the name). 2-D uint8 arrays travel as lossless PNG, `.xyt` templates as
their text. `google.protobuf.Value` stores every number as a double, so decoders cast ints
back explicitly; doubles survive exactly. `AttemptConfig` travels as `attempt_no` plus
`configuration_id`, and is looked up in the receiver's (identical, frozen) config; a mismatch
is `invalid_input`. This keeps `sbmex_overrides` ints from turning into floats.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from fpv.adapter import XytTemplate, parse_xyt

if TYPE_CHECKING:
    from a2a.types import Part

    from fpv.config import AttemptConfig, PipelineConfig
    from fpv.extract import ExtractionResult
    from fpv.matcher import MatchResult
    from fpv.preprocess import PreprocessResult
    from fpv.quality import QualityResult

PNG = "image/png"
XYT = "text/plain"
SAMPLE = "application/octet-stream"

# Same fields as pyfing.definitions.Minutia (a namedtuple), so values compare equal to phase 1's.
Minutia = collections.namedtuple("Minutia", "x y direction type quality")


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NFIQ2_TOOL_ERROR = "nfiq2_tool_error"
    INSUFFICIENT_MINUTIAE = "insufficient_minutiae"
    BOZORTH3_TOOL_ERROR = "bozorth3_tool_error"
    STAGE_ERROR = "stage_error"


class InvalidInput(ValueError):
    """A request does not decode into the skill's input (Req 2.5)."""


class RemoteStageError(Exception):
    """A remote stage failed with a code that has no phase 1 exception (or was unreachable)."""

    def __init__(self, message: str, code: str = ErrorCode.STAGE_ERROR, error_type: str | None = None) -> None:
        super().__init__(message)
        self.code, self.error_type = code, error_type


@dataclass(slots=True)
class Payload:
    data: dict[str, Any] = field(default_factory=dict)
    files: dict[str, tuple[bytes, str]] = field(default_factory=dict)  # name -> (bytes, media type)


# --------------------------------------------------------------------------- parts


def to_parts(p: Payload) -> list[Part]:
    from a2a.helpers.proto_helpers import new_data_part, new_raw_part

    parts = [new_raw_part(raw, media, filename=name) for name, (raw, media) in p.files.items()]
    return [new_data_part(p.data), *parts]


def _whole(v: Any) -> Any:
    """Whole-number doubles back to int (decoders cast float fields explicitly, so this is lossless)."""
    if isinstance(v, float) and v.is_integer() and abs(v) < 2**53:
        return int(v)
    if isinstance(v, dict):
        return {k: _whole(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_whole(x) for x in v]
    return v


def from_parts(parts: Any) -> Payload:
    from google.protobuf.json_format import MessageToDict

    out = Payload()
    for part in parts:
        kind = part.WhichOneof("content")
        if kind == "data":
            value = _whole(MessageToDict(part.data))
            if not isinstance(value, dict):
                raise InvalidInput("data part is not a JSON object")
            out.data.update(value)
        elif kind == "raw":
            out.files[part.filename] = (bytes(part.raw), part.media_type)
        else:
            raise InvalidInput(f"unsupported part kind {kind!r}")
    return out


# --------------------------------------------------------------------------- field helpers


def _int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
        raise InvalidInput(f"expected an integer, got {v!r}")
    return int(v)


def _opt_int(v: Any) -> int | None:
    return None if v is None else _int(v)


def _get(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise InvalidInput(f"missing field {key!r}")
    return d[key]


def _file(p: Payload, name: str, media: str) -> bytes:
    if name not in p.files:
        raise InvalidInput(f"missing raw part {name!r}")
    raw, got = p.files[name]
    if got != media:
        raise InvalidInput(f"raw part {name!r} has media type {got!r}, expected {media!r}")
    return raw


def _png(p: Payload, name: str) -> np.ndarray:
    from fpv import imaging

    try:
        return imaging.from_png_bytes(_file(p, name, PNG))
    except imaging.InputImageError as exc:
        raise InvalidInput(f"raw part {name!r}: {exc}") from exc


# --------------------------------------------------------------------------- contracts


def encode_image(img: np.ndarray, name: str = "image") -> Payload:
    from fpv import imaging

    return Payload(files={name: (imaging.png_bytes(img), PNG)})


def decode_image(p: Payload, name: str = "image") -> np.ndarray:
    return _png(p, name)


def encode_attempt(attempt: AttemptConfig, configuration_id: str) -> dict[str, Any]:
    return {"attempt_no": attempt.attempt_no, "configuration_id": configuration_id}


def decode_attempt(d: dict[str, Any], conf: PipelineConfig) -> tuple[AttemptConfig, str]:
    no, cid = _int(_get(d, "attempt_no")), _get(d, "configuration_id")
    try:
        attempt = conf.attempt(no)
    except KeyError as exc:
        raise InvalidInput(str(exc)) from exc
    if conf.configuration_id(attempt) != cid:
        raise InvalidInput(f"configuration_id {cid!r} does not match attempt {no} of {conf.config_set_id}")
    return attempt, cid


def encode_quality(q: QualityResult) -> Payload:
    return Payload({"score": q.score, "actionable_feedback": dict(q.actionable_feedback), "raw_output": q.raw_output,
                    "tool_version": q.tool_version, "duration_ms": q.duration_ms, "upscale": q.upscale,
                    "error": q.error, "scored_shape": list(q.scored_shape)})


def decode_quality(p: Payload) -> QualityResult:
    from fpv.quality import QualityResult

    d = p.data
    return QualityResult(score=_opt_int(d.get("score")),
                         actionable_feedback={k: float(v) for k, v in _get(d, "actionable_feedback").items()},
                         raw_output=_get(d, "raw_output"), tool_version=_get(d, "tool_version"),
                         duration_ms=float(_get(d, "duration_ms")), upscale=float(_get(d, "upscale")),
                         error=d.get("error"), scored_shape=tuple(_int(v) for v in _get(d, "scored_shape")))


def encode_preprocess(pre: PreprocessResult) -> Payload:
    from fpv import imaging

    return Payload(
        {**encode_attempt(pre.attempt, pre.configuration_id), "scale_factor": pre.scale_factor,
         "original_shape": list(pre.original_shape), "timings_ms": dict(pre.timings_ms)},
        {"enhanced": (imaging.png_bytes(pre.enhanced), PNG), "mask": (imaging.png_bytes(pre.mask), PNG)})


def decode_preprocess(p: Payload, conf: PipelineConfig) -> PreprocessResult:
    from fpv.preprocess import PreprocessResult

    d = p.data
    attempt, cid = decode_attempt(d, conf)
    return PreprocessResult(attempt=attempt, enhanced=_png(p, "enhanced"), mask=_png(p, "mask"),
                            scale_factor=float(_get(d, "scale_factor")),
                            original_shape=tuple(_int(v) for v in _get(d, "original_shape")),
                            configuration_id=cid,
                            timings_ms={k: float(v) for k, v in _get(d, "timings_ms").items()})


def encode_extraction(ex: ExtractionResult) -> Payload:
    return Payload({"minutiae": [[m.x, m.y, m.direction, m.type, m.quality] for m in ex.minutiae],
                    "count": ex.count, "configuration_id": ex.configuration_id, "duration_ms": ex.duration_ms})


def decode_extraction(p: Payload) -> ExtractionResult:
    from fpv.extract import ExtractionResult

    d = p.data
    try:
        minutiae = tuple(Minutia(_int(x), _int(y), float(a), str(t), float(q)) for x, y, a, t, q in _get(d, "minutiae"))
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"invalid minutiae: {exc}") from exc
    return ExtractionResult(minutiae=minutiae, count=_int(_get(d, "count")),
                            configuration_id=_get(d, "configuration_id"), duration_ms=float(_get(d, "duration_ms")))


def encode_xyt(t: XytTemplate, name: str = "template") -> Payload:
    return Payload({f"{name}_count": t.count, "adapter_version": t.adapter_version},
                   {name: (t.to_text().encode("ascii"), XYT)})


def decode_xyt(p: Payload, name: str = "template") -> XytTemplate:
    try:
        tpl = parse_xyt(_file(p, name, XYT).decode("ascii"), source=name)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidInput(f"raw part {name!r}: {exc}") from exc
    version = p.data.get("adapter_version", tpl.adapter_version)
    return XytTemplate(tpl.minutiae, adapter_version=version)


def encode_match(m: MatchResult) -> Payload:
    return Payload({"score": m.score, "probe_count": m.probe_count, "gallery_count": m.gallery_count,
                    "tool_version": m.tool_version, "duration_ms": m.duration_ms})


def decode_match(p: Payload) -> MatchResult:
    from fpv.matcher import MatchResult

    d = p.data
    return MatchResult(score=_int(_get(d, "score")), probe_count=_int(_get(d, "probe_count")),
                       gallery_count=_int(_get(d, "gallery_count")), tool_version=_get(d, "tool_version"),
                       duration_ms=float(_get(d, "duration_ms")))


# --------------------------------------------------------------------------- errors


def encode_error(exc: BaseException) -> dict[str, Any]:
    """Error data part for a failed (or rejected) task (Req 1.3)."""
    from fpv.matcher import InsufficientMinutiae, MatcherError
    from fpv.quality import QualityToolError

    out: dict[str, Any] = {"error_type": type(exc).__name__, "error": str(exc)[:500]}
    if isinstance(exc, InvalidInput):
        out["error_code"] = ErrorCode.INVALID_INPUT
    elif isinstance(exc, QualityToolError):
        out["error_code"] = ErrorCode.NFIQ2_TOOL_ERROR
    elif isinstance(exc, InsufficientMinutiae):
        out.update(error_code=ErrorCode.INSUFFICIENT_MINUTIAE, which=exc.which, count=exc.count, minimum=exc.minimum)
    elif isinstance(exc, MatcherError):
        out["error_code"] = ErrorCode.BOZORTH3_TOOL_ERROR
    else:
        out["error_code"] = ErrorCode.STAGE_ERROR
    out["error_code"] = str(out["error_code"])
    return out


def decode_error(d: dict[str, Any]) -> Exception:
    """The phase 1 exception for an error data part, so the attempt loop and controller behave as in phase 1."""
    from fpv.matcher import InsufficientMinutiae, MatcherError
    from fpv.quality import QualityToolError

    code, msg = d.get("error_code"), d.get("error", "")
    if code == ErrorCode.NFIQ2_TOOL_ERROR:
        return QualityToolError(msg)
    if code == ErrorCode.INSUFFICIENT_MINUTIAE:
        return InsufficientMinutiae(d["which"], _int(d["count"]), _int(d.get("minimum", 10)))
    if code == ErrorCode.BOZORTH3_TOOL_ERROR:
        return MatcherError(msg)
    name = d.get("error_type")
    # Same class name as the remote exception, so `AttemptRecord.error` ("<Type>: msg") matches phase 1.
    cls = _remote_class(name) if isinstance(name, str) and name.isidentifier() else RemoteStageError
    return cls(msg, code or ErrorCode.STAGE_ERROR, name)


_REMOTE_CLASSES: dict[str, type[RemoteStageError]] = {}


def _remote_class(name: str) -> type[RemoteStageError]:
    if name not in _REMOTE_CLASSES:
        _REMOTE_CLASSES[name] = type(name, (RemoteStageError,), {})
    return _REMOTE_CLASSES[name]

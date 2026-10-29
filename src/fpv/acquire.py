"""The attempt loop shared by `fpv extract` and the pipeline controller (Req 5.3, 5.4, 5.7, 6.3).

`acquire_template(image, config, stages)` runs, for attempt k = 1..max_attempts on the raw
sample: preprocess -> quality_after -> extract -> adapt, and stops at the first attempt whose
*adapted* (deduplicated) `.xyt` count reaches `thresholds.minutiae_min` (Req 6.3). The raw
NFIQ 2 gate is not part of this loop; callers apply it first (Req 4.1).

Failure policy (design "Error Handling"):

* an exception in preprocess / extract / adapt fails that attempt only; the loop continues;
* `QualityToolError` (NFIQ 2 missing, timeout, crash) propagates: it is a tool error, not a
  property of the attempt; an unscorable enhanced image is just `quality_after=None` (Req 5.6);
* after the last attempt: `REJECTED_MINUTIAE / insufficient_minutiae` if any attempt completed,
  `ERROR / preprocess_error` if every attempt raised.

`span(stage, attempt_no, configuration_id)` is an optional context-manager factory wrapped
around every stage call; it yields a dict the loop fills with stage details. The controller
(Task 9) uses it to write `stage_events`; exceptions pass through it before being handled here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from fpv.adapter import ADAPTER_VERSION, XytTemplate, to_xyt
from fpv.types import Status

if TYPE_CHECKING:
    from fpv.config import AttemptConfig, PipelineConfig
    from fpv.extract import ExtractionResult
    from fpv.preprocess import PreprocessResult
    from fpv.quality import QualityResult

__all__ = [
    "Acquired",
    "AcquisitionRejected",
    "AcquisitionStages",
    "AttemptRecord",
    "LocalAcquisitionStages",
    "Span",
    "acquire_template",
    "adapt",
]

REASON_INSUFFICIENT_MINUTIAE = "insufficient_minutiae"
REASON_PREPROCESS_ERROR = "preprocess_error"

Span = Callable[[str, int, str], AbstractContextManager[dict[str, Any]]]


class AcquisitionStages(Protocol):
    """The subset of the pipeline `Stages` protocol the attempt loop needs."""

    def preprocess(self, image: np.ndarray, attempt: AttemptConfig, configuration_id: str) -> PreprocessResult: ...
    def quality_after(self, pre: PreprocessResult) -> QualityResult: ...
    def extract(self, pre: PreprocessResult) -> ExtractionResult: ...
    def adapt(self, ex: ExtractionResult, pre: PreprocessResult) -> XytTemplate: ...


def adapt(ex: ExtractionResult, pre: PreprocessResult) -> XytTemplate:
    """Adapter stage: undo `pre.scale_factor` and map to the original sample's `.xyt` frame."""
    return to_xyt(ex.minutiae, pre.original_shape, pre.scale_factor)


class LocalAcquisitionStages:
    """In-process implementations (pyfing, NFIQ 2 CLI, adapter); heavy imports are lazy."""

    def __init__(self, config: PipelineConfig) -> None:
        from fpv.quality import QualitySettings

        self._quality = QualitySettings.from_config(config)

    def preprocess(self, image: np.ndarray, attempt: AttemptConfig, configuration_id: str) -> PreprocessResult:
        from fpv.preprocess import preprocess

        return preprocess(image, attempt, configuration_id=configuration_id)

    def quality_after(self, pre: PreprocessResult) -> QualityResult:
        from fpv.preprocess import quality_after

        return quality_after(pre, self._quality)

    def extract(self, pre: PreprocessResult) -> ExtractionResult:
        from fpv.extract import extract

        return extract(pre)

    def adapt(self, ex: ExtractionResult, pre: PreprocessResult) -> XytTemplate:
        return adapt(ex, pre)


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_no: int
    configuration_id: str
    quality_after: int | None          # None: unscorable (Req 5.6) or the attempt failed before scoring
    raw_count: int | None              # SBMEX count; None if extraction did not run
    minutiae_count: int | None         # adapted (deduplicated) count; None if adaptation did not run
    error: str | None = None           # "<ExceptionType>: message" when the attempt raised


@dataclass(frozen=True, slots=True)
class Acquired:
    template: XytTemplate
    preprocess: PreprocessResult
    extraction: ExtractionResult
    quality_after: QualityResult
    attempts: tuple[AttemptRecord, ...]

    @property
    def attempt_no(self) -> int:
        return self.attempts[-1].attempt_no

    @property
    def configuration_id(self) -> str:
        return self.attempts[-1].configuration_id

    @property
    def scale_factor(self) -> float:
        return self.preprocess.scale_factor


@dataclass(frozen=True, slots=True)
class AcquisitionRejected:
    status: Status                     # REJECTED_MINUTIAE or ERROR
    reason: str                        # insufficient_minutiae | preprocess_error
    attempts: tuple[AttemptRecord, ...]
    template: XytTemplate | None = None  # last completed attempt's template (diagnostic only)


@contextlib.contextmanager
def _no_span(stage: str, attempt_no: int, configuration_id: str) -> Iterator[dict[str, Any]]:
    yield {}


def acquire_template(image: np.ndarray, config: PipelineConfig, stages: AcquisitionStages | None = None,
                     *, span: Span | None = None) -> Acquired | AcquisitionRejected:
    """Run the bounded attempt policy on the raw sample `image` (Req 5.1-5.4, 5.7, 6.3)."""
    if config.adapter_version != ADAPTER_VERSION:
        raise ValueError(f"config adapter.version {config.adapter_version!r} does not match the "
                         f"adapter in use ({ADAPTER_VERSION!r}); configuration_ids would be wrong")
    if stages is None:
        stages = LocalAcquisitionStages(config)
    span = span or _no_span
    from fpv.quality import QualityToolError

    minimum = config.thresholds.minutiae_min
    records: list[AttemptRecord] = []
    last_template: XytTemplate | None = None

    for attempt in config.attempts[: config.max_attempts]:
        k = attempt.attempt_no
        cid = config.configuration_id(attempt)
        qa_score: int | None = None
        raw_count: int | None = None
        try:
            with span("preprocess", k, cid) as d:
                pre = stages.preprocess(image, attempt, cid)
                d.update(timings_ms=dict(pre.timings_ms), scale_factor=pre.scale_factor)
            with span("quality_after", k, cid) as d:
                qa = stages.quality_after(pre)
                qa_score = qa.score
                d.update(score=qa.score, feedback=dict(qa.actionable_feedback))
            with span("extract", k, cid) as d:
                ex = stages.extract(pre)
                raw_count = ex.count
                d.update(raw_count=ex.count)
            with span("adapt", k, cid) as d:
                tpl = stages.adapt(ex, pre)
                d.update(raw_count=ex.count, dedup_removed=ex.count - tpl.count, count=tpl.count,
                         minutiae_min=minimum, passed=tpl.count >= minimum)
        except QualityToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed attempt moves on to the next one
            records.append(AttemptRecord(k, cid, qa_score, raw_count, None, f"{type(exc).__name__}: {exc}"))
            continue

        records.append(AttemptRecord(k, cid, qa_score, raw_count, tpl.count))
        last_template = tpl
        if tpl.count >= minimum:
            return Acquired(template=tpl, preprocess=pre, extraction=ex, quality_after=qa,
                            attempts=tuple(records))

    completed = any(r.error is None for r in records)
    if completed:
        return AcquisitionRejected(Status.REJECTED_MINUTIAE, REASON_INSUFFICIENT_MINUTIAE, tuple(records),
                                   last_template)
    return AcquisitionRejected(Status.ERROR, REASON_PREPROCESS_ERROR, tuple(records))

"""Pipeline controller: `enroll` and `verify` (Req 4, 5, 8, 9, 11, 15.2).

One run = one sample processed for one operation. Both operations share the same acquisition
path (Req 4.7, 5.7):

    load -> quality_before (NFIQ 2 gate) -> [preprocess -> quality_after -> extract -> adapt] x k

then enrollment persists the template (`persist`), and verification stores the probe
(`persist`), looks up the active template (`lookup`), runs Bozorth3 (`match`) and records the
decision (`decide`). Every stage invocation is written to `stage_events` by `_stage`, with a
monotonic duration, a UTC `started_at` and a per-run `seq`; a stage that raises is logged with
outcome `error` and the exception type before the run ends (Req 11.3-11.5).

Errors map to run statuses per the design's Error Handling table, and every run is finished
with `ended_at`, including on unexpected exceptions (`ERROR / unexpected_error`). Ground-truth
labels are only copied into `verifications.ground_truth_genuine`; they never influence a gate
or a decision (Req 9.10). The controller never prompts or prints (Req 15.2); the CLI owns
sessions and presentation limits. Stage implementations are injected through `Stages`, so tests
(and phase 2 agents) can replace every stage.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from fpv.acquire import (
    REASON_INSUFFICIENT_MINUTIAE,
    REASON_PREPROCESS_ERROR,
    Acquired,
    AcquisitionRejected,
    AcquisitionStages,
    AttemptRecord,
    LocalAcquisitionStages,
    acquire_template,
)
from fpv.adapter import XytTemplate
from fpv.config import pipeline_version as _pipeline_version
from fpv.imaging import InputImageError, load_gray8
from fpv.matcher import MIN_BOZORTH_MINUTIAE, InsufficientMinutiae, MatcherError, MatchResult, decide
from fpv.quality import REASON_BELOW_THRESHOLD, REASON_NFIQ2_FAILED, QualityToolError, quality_gate
from fpv.storage import (
    Operation,
    Repository,
    RunStart,
    StageEvent,
    TemplateMeta,
    VerificationRow,
    quality_after_entries,
    utc_now_iso,
)
from fpv.types import Identity, Status

if TYPE_CHECKING:
    from fpv.catalog import Catalog, SampleRecord
    from fpv.config import PipelineConfig
    from fpv.quality import QualityResult

__all__ = [
    "AttemptRecord",
    "CachedProbe",
    "Clock",
    "EnrollmentOutcome",
    "LocalStages",
    "PipelineController",
    "ProbeCache",
    "Stages",
    "SystemClock",
    "VerificationOutcome",
]

REASON_UNREADABLE_IMAGE = "unreadable_image"
REASON_NFIQ2_TOOL_ERROR = "nfiq2_tool_error"
REASON_PROBE_BELOW_BOZORTH_MIN = "probe_below_bozorth_min"
REASON_REFERENCE_BELOW_BOZORTH_MIN = "reference_below_bozorth_min"
REASON_BOZORTH3_TOOL_ERROR = "bozorth3_tool_error"
REASON_PERSIST_ERROR = "persist_error"
REASON_UNEXPECTED_ERROR = "unexpected_error"
REASON_INTERRUPTED = "interrupted"

STAGE_NAMES = ("load", "quality_before", "preprocess", "quality_after", "extract", "adapt",
               "persist", "lookup", "match", "decide")


# --------------------------------------------------------------------------- injected dependencies


class Stages(AcquisitionStages, Protocol):
    """Every stage the controller calls; the attempt-loop stages come from `AcquisitionStages`."""

    def assess_quality(self, image: np.ndarray) -> QualityResult: ...
    def match(self, probe: Path, gallery: Path) -> MatchResult: ...


class LocalStages(LocalAcquisitionStages):
    """In-process stages: NFIQ 2 CLI, pyfing, SBMEX, the `.xyt` adapter and the Bozorth3 CLI."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._nbis_version = config.tools.get("nbis", "unknown")

    def assess_quality(self, image: np.ndarray) -> QualityResult:
        from fpv.quality import assess_quality

        return assess_quality(image, self._quality)

    def match(self, probe: Path, gallery: Path) -> MatchResult:
        from fpv.matcher import match

        return match(probe, gallery, tool_version=self._nbis_version)


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...
    def utc_now_iso(self) -> str: ...


class SystemClock:
    """`perf_counter_ns` for durations (monotonic, Req 11.4) and UTC ISO-8601 for timestamps."""

    def monotonic_ns(self) -> int:
        return time.perf_counter_ns()

    def utc_now_iso(self) -> str:
        return utc_now_iso()


@dataclass(frozen=True, slots=True)
class CachedProbe:
    """A processed probe, reusable across one probe's evaluation trials (Req 13.6, Task 11).

    `template is None` means acquisition was rejected with `status` / `reason`. Tool errors are
    never cached.
    """

    quality_before: int | None
    attempts: tuple[AttemptRecord, ...]
    template: XytTemplate | None
    attempt_no: int | None = None
    configuration_id: str | None = None
    status: Status | None = None
    reason: str | None = None
    message: str = ""


class ProbeCache(Protocol):
    """Keyed by `(sample sha256, config_set_id)`; only `evaluate.py` injects one."""

    def get(self, key: tuple[str, str]) -> CachedProbe | None: ...
    def put(self, key: tuple[str, str], value: CachedProbe) -> None: ...


# --------------------------------------------------------------------------- outcomes


@dataclass(frozen=True, slots=True)
class EnrollmentOutcome:
    run_id: str
    status: Status
    reason: str | None
    message: str
    quality_before: int | None
    attempts: tuple[AttemptRecord, ...]
    template_id: str | None
    template_version: int | None


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    run_id: str
    status: Status
    reason: str | None
    message: str
    quality_before: int | None
    attempts: tuple[AttemptRecord, ...]
    template_id: str | None
    score: int | None
    threshold: int
    ground_truth_genuine: bool | None


@dataclass(slots=True)
class _StageCtx:
    outcome: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Run:
    """Mutable per-run state; turned into an immutable outcome when the run ends."""

    run_id: str
    seq: int = 0
    sample: SampleRecord | None = None
    quality_before: int | None = None
    attempts: tuple[AttemptRecord, ...] = ()
    attempts_started: int = 0
    template_id: str | None = None
    template_version: int | None = None
    score: int | None = None
    ground_truth_genuine: bool | None = None
    status: Status | None = None
    reason: str | None = None
    message: str = ""

    def end(self, status: Status, reason: str | None, message: str) -> None:
        self.status, self.reason, self.message = status, reason, message

    @property
    def attempts_used(self) -> int:
        return max(len(self.attempts), self.attempts_started)


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


# --------------------------------------------------------------------------- controller


class PipelineController:
    def __init__(self, config: PipelineConfig, repo: Repository, catalog: Catalog,
                 stages: Stages | None = None, clock: Clock | None = None,
                 probe_cache: ProbeCache | None = None, *, pipeline_version: str | None = None) -> None:
        self._conf = config
        self._repo = repo
        self._catalog = catalog
        self._stages: Stages = stages if stages is not None else LocalStages(config)
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._probe_cache = probe_cache
        self.pipeline_version = pipeline_version or _pipeline_version()

    @property
    def config(self) -> PipelineConfig:
        return self._conf

    # ----------------------------------------------------------------- public operations

    def enroll(self, sample_path: Path | str, identity: Identity, *, session_id: str | None = None,
               presentation_no: int = 1) -> EnrollmentOutcome:
        """Acquire a template from `sample_path` and persist it as `identity`'s active template."""
        run = self._run("enroll", sample_path, identity, session_id, presentation_no,
                        lambda r: self._enroll_body(r, Path(sample_path), identity))
        return EnrollmentOutcome(
            run_id=run.run_id, status=run.status, reason=run.reason, message=run.message,
            quality_before=run.quality_before, attempts=run.attempts,
            template_id=run.template_id, template_version=run.template_version)

    def verify(self, sample_path: Path | str, identity: Identity, *, session_id: str | None = None,
               presentation_no: int = 1) -> VerificationOutcome:
        """Acquire a probe from `sample_path` and compare it with `identity`'s active template."""
        run = self._run("verify", sample_path, identity, session_id, presentation_no,
                        lambda r: self._verify_body(r, Path(sample_path), identity))
        return VerificationOutcome(
            run_id=run.run_id, status=run.status, reason=run.reason, message=run.message,
            quality_before=run.quality_before, attempts=run.attempts, template_id=run.template_id,
            score=run.score, threshold=self._conf.thresholds.bozorth3_match,
            ground_truth_genuine=run.ground_truth_genuine)

    # ----------------------------------------------------------------- run lifecycle

    def _run(self, operation: Operation, sample_path: Path | str, identity: Identity,
             session_id: str | None, presentation_no: int, body: Callable[[_Run], None]) -> _Run:
        run = _Run(run_id=str(uuid.uuid4()))
        self._repo.start_run(RunStart(
            run_id=run.run_id, session_id=session_id or str(uuid.uuid4()),
            presentation_no=presentation_no, operation=operation, identity=identity, sample_id=None,
            pipeline_version=self.pipeline_version, config_set_id=self._conf.config_set_id,
            started_at=self._clock.utc_now_iso(), control_mode="controller", orchestrator_vision="off"))
        finished = False
        try:
            try:
                body(run)
            except Exception as exc:  # noqa: BLE001 - the controller boundary (design: Error Handling)
                run.end(Status.ERROR, REASON_UNEXPECTED_ERROR, f"Unexpected error: {_error_text(exc)}")
            if run.status is None:  # pragma: no cover - every body path ends the run
                run.end(Status.ERROR, REASON_UNEXPECTED_ERROR, "Run ended without a status")
            self._finish(run)
            finished = True
        finally:
            if not finished:  # KeyboardInterrupt etc.: still never leave a run without ended_at
                run.end(Status.ERROR, REASON_INTERRUPTED, "Run interrupted")
                with contextlib.suppress(Exception):
                    self._finish(run)
        return run

    def _finish(self, run: _Run) -> None:
        assert run.status is not None
        self._repo.finish_run(run.run_id, run.status, run.reason, run.attempts_used,
                              ended_at=self._clock.utc_now_iso())

    @contextlib.contextmanager
    def _stage(self, run: _Run, stage: str, attempt_no: int | None = None,
               configuration_id: str | None = None) -> Iterator[_StageCtx]:
        """Time one stage invocation and write its `stage_events` row (Req 11.3-11.5)."""
        ctx = _StageCtx()
        started_at = self._clock.utc_now_iso()
        t0 = self._clock.monotonic_ns()
        try:
            yield ctx
        except BaseException as exc:
            details = {**ctx.details, "error_type": type(exc).__name__, "error": str(exc)[:500]}
            self._log(run, stage, attempt_no, configuration_id, started_at, t0, "error", details)
            raise
        self._log(run, stage, attempt_no, configuration_id, started_at, t0, ctx.outcome, ctx.details)

    def _log(self, run: _Run, stage: str, attempt_no: int | None, configuration_id: str | None,
             started_at: str, t0: int, outcome: str, details: dict[str, Any]) -> None:
        duration_ms = (self._clock.monotonic_ns() - t0) / 1e6
        run.seq += 1
        self._repo.log_stage_event(StageEvent(
            run_id=run.run_id, seq=run.seq, stage=stage, started_at=started_at,
            duration_ms=duration_ms, outcome=outcome, attempt_no=attempt_no,
            configuration_id=configuration_id, details=details))

    def _span(self, run: _Run) -> Callable[[str, int, str], contextlib.AbstractContextManager[dict[str, Any]]]:
        """Adapter from `acquire_template`'s span hook to `_stage`."""

        @contextlib.contextmanager
        def span(stage: str, attempt_no: int, configuration_id: str) -> Iterator[dict[str, Any]]:
            if stage == "preprocess":
                run.attempts_started = max(run.attempts_started, attempt_no)
            with self._stage(run, stage, attempt_no, configuration_id) as st:
                yield st.details
                if st.details.get("passed") is False:
                    st.outcome = "rejected"

        return span

    # ----------------------------------------------------------------- shared stages

    def _load(self, run: _Run, path: Path) -> tuple[np.ndarray, SampleRecord] | None:
        try:
            with self._stage(run, "load") as st:
                rec = self._catalog.record(path)
                image = load_gray8(path)
                self._repo.upsert_sample(rec)
                self._repo.set_run_sample(run.run_id, rec.sample_id)
                st.details.update(sample_id=rec.sample_id, source_path=rec.source_path, sha256=rec.sha256,
                                  width=rec.width, height=rec.height)
        except InputImageError as exc:
            run.end(Status.REJECTED_QUALITY, REASON_UNREADABLE_IMAGE, f"Sample rejected: cannot read the image ({exc})")
            return None
        run.sample = rec
        return image, rec

    def _acquire(self, run: _Run, image: np.ndarray, *, cache_hit: bool | None = None) -> Acquired | None:
        """Raw NFIQ 2 gate, then the bounded attempt loop (identical for both operations, Req 4.7, 5.7).

        Returns the acquisition, or None after ending the run with a rejection or error.
        `cache_hit` (False on a probe-cache miss) is recorded in the `quality_before` details.
        """
        threshold = self._conf.thresholds.nfiq2_min
        try:
            with self._stage(run, "quality_before") as st:
                if cache_hit is not None:
                    st.details["cache_hit"] = cache_hit
                q = self._stages.assess_quality(image)
                gate = quality_gate(q, threshold)
                run.quality_before = q.score
                st.details.update(score=q.score, feedback=dict(q.actionable_feedback), threshold=threshold,
                                  passed=gate.passed, reason=gate.reason)
                if getattr(q, "error", None):
                    st.details["nfiq2_error"] = q.error
                if not gate.passed:
                    st.outcome = "rejected"
        except QualityToolError as exc:
            run.end(Status.ERROR, REASON_NFIQ2_TOOL_ERROR, f"NFIQ 2 failed: {exc}")
            return None
        if not gate.passed:
            if gate.reason == REASON_BELOW_THRESHOLD:
                msg = (f"Sample rejected: NFIQ 2 quality {gate.score} is below the minimum {threshold}; "
                       "please present a new sample")
            else:
                msg = "Sample rejected: NFIQ 2 could not score the sample; please present a new sample"
            run.end(Status.REJECTED_QUALITY, gate.reason or REASON_NFIQ2_FAILED, msg)
            return None

        try:
            result = acquire_template(image, self._conf, self._stages, span=self._span(run))
        except QualityToolError as exc:
            run.end(Status.ERROR, REASON_NFIQ2_TOOL_ERROR, f"NFIQ 2 failed: {exc}")
            return None
        run.attempts = result.attempts
        if isinstance(result, AcquisitionRejected):
            if result.reason == REASON_INSUFFICIENT_MINUTIAE:
                msg = (f"Sample rejected: fewer than {self._conf.thresholds.minutiae_min} minutiae after "
                       f"{len(result.attempts)} attempt(s); please present a new sample")
            elif result.reason == REASON_PREPROCESS_ERROR:
                msg = "Processing failed: every preprocessing attempt raised an error"
            else:  # pragma: no cover - acquire_template has no other reasons
                msg = f"Sample rejected: {result.reason}"
            run.end(result.status, result.reason, msg)
            return None
        return result

    # ----------------------------------------------------------------- enrollment

    def _enroll_body(self, run: _Run, path: Path, identity: Identity) -> None:
        loaded = self._load(run, path)
        if loaded is None:
            return
        image, rec = loaded
        acq = self._acquire(run, image)
        if acq is None:
            return
        assert run.quality_before is not None  # the gate passed
        meta = TemplateMeta(
            source_sample_id=rec.sample_id, quality_before=run.quality_before,
            quality_after=quality_after_entries(acq.attempts), scale_factor=acq.scale_factor,
            pipeline_version=self.pipeline_version, configuration_id=acq.configuration_id)
        try:
            with self._stage(run, "persist", acq.attempt_no, acq.configuration_id) as st:
                row = self._repo.enroll_template(identity, acq.template, meta)
                st.details.update(kind="template", template_id=row.template_id,
                                  template_version=row.template_version, minutiae_count=row.minutiae_count,
                                  location=row.template_location)
        except Exception as exc:  # noqa: BLE001 - rolled back by the repository (Req 8.6)
            run.end(Status.ERROR, REASON_PERSIST_ERROR, f"Enrollment failed: could not persist the template "
                                                        f"({_error_text(exc)})")
            return
        run.template_id, run.template_version = row.template_id, row.template_version
        run.end(Status.ENROLLED, None, f"Enrollment successful (template v{row.template_version})")

    # ----------------------------------------------------------------- verification

    def _probe(self, run: _Run, image: np.ndarray, rec: SampleRecord) -> tuple[XytTemplate, int, str] | None:
        """Acquire the probe, through the optional probe cache (Req 13.6)."""
        key = (rec.sha256, self._conf.config_set_id)
        if self._probe_cache is not None:
            hit = self._probe_cache.get(key)
            if hit is not None:
                return self._replay_cached(run, hit)
        acq = self._acquire(run, image, cache_hit=False if self._probe_cache is not None else None)
        if self._probe_cache is not None:
            if acq is not None:
                self._probe_cache.put(key, CachedProbe(run.quality_before, acq.attempts, acq.template,
                                                       acq.attempt_no, acq.configuration_id))
            elif run.status in (Status.REJECTED_QUALITY, Status.REJECTED_MINUTIAE):
                self._probe_cache.put(key, CachedProbe(run.quality_before, run.attempts, None,
                                                       status=run.status, reason=run.reason,
                                                       message=run.message))
        if acq is None:
            return None
        return acq.template, acq.attempt_no, acq.configuration_id

    def _replay_cached(self, run: _Run, hit: CachedProbe) -> tuple[XytTemplate, int, str] | None:
        with self._stage(run, "quality_before") as st:
            st.outcome = "skipped"
            st.details.update(cache_hit=True, score=hit.quality_before, attempts=len(hit.attempts),
                              status=hit.status.value if hit.status else None)
        run.quality_before = hit.quality_before
        run.attempts = hit.attempts
        if hit.template is None:
            run.end(hit.status or Status.ERROR, hit.reason, hit.message)
            return None
        assert hit.attempt_no is not None and hit.configuration_id is not None
        return hit.template, hit.attempt_no, hit.configuration_id

    def _verify_body(self, run: _Run, path: Path, identity: Identity) -> None:
        loaded = self._load(run, path)
        if loaded is None:
            return
        image, rec = loaded
        # Ground truth is recorded only; it never feeds a gate or decision (Req 9.9, 9.10).
        genuine = None if rec.label is None else rec.label.identity == identity
        run.ground_truth_genuine = genuine

        probe = self._probe(run, image, rec)
        if probe is None:
            return
        template, attempt_no, cid = probe

        try:
            with self._stage(run, "persist", attempt_no, cid) as st:
                probe_path = self._repo.write_probe(run.run_id, template)
                st.details.update(kind="probe", minutiae_count=template.count,
                                  location=str(probe_path))
        except Exception as exc:  # noqa: BLE001
            run.end(Status.ERROR, REASON_PERSIST_ERROR, f"Verification failed: could not store the probe "
                                                        f"({_error_text(exc)})")
            return

        with self._stage(run, "lookup") as st:
            reference = self._repo.get_active_template(identity)
            st.details["found"] = reference is not None
            if reference is not None:
                st.details.update(template_id=reference.template_id,
                                  template_version=reference.template_version)
        if reference is None:
            run.end(Status.NOT_ENROLLED, None, "Not enrolled")
            return
        run.template_id = reference.template_id

        try:
            with self._stage(run, "match") as st:
                result = self._stages.match(probe_path, self._repo.resolve_location(reference.template_location))
                run.score = result.score
                st.details.update(score=result.score, probe_count=result.probe_count,
                                  gallery_count=result.gallery_count, tool_version=result.tool_version)
        except InsufficientMinutiae as exc:
            if exc.which == "probe":
                run.end(Status.REJECTED_MINUTIAE, REASON_PROBE_BELOW_BOZORTH_MIN,
                        f"Sample rejected: the probe has {exc.count} minutiae, Bozorth3 needs at least "
                        f"{MIN_BOZORTH_MINUTIAE}; please present a new sample")
            else:
                run.end(Status.ERROR, REASON_REFERENCE_BELOW_BOZORTH_MIN,
                        f"The enrolled template has {exc.count} minutiae, below the Bozorth3 minimum of "
                        f"{MIN_BOZORTH_MINUTIAE}")
            return
        except MatcherError as exc:
            run.end(Status.ERROR, REASON_BOZORTH3_TOOL_ERROR, f"Bozorth3 failed: {exc}")
            return

        threshold = self._conf.thresholds.bozorth3_match
        with self._stage(run, "decide") as st:
            decision = decide(result.score, threshold)
            self._repo.record_verification(VerificationRow(
                run_id=run.run_id, template_id=reference.template_id, probe_location=probe_path,
                score=result.score, threshold=threshold, decision=decision, ground_truth_genuine=genuine))
            st.details.update(score=result.score, threshold=threshold, decision=decision.value)
        run.end(decision, None, "Match" if decision is Status.MATCH else "No match")


# --------------------------------------------------------------------------- demo


def format_trail(events: Sequence[dict[str, Any]]) -> list[str]:
    """One line per stage event: seq, stage, attempt, outcome, duration."""
    lines = []
    for ev in events:
        attempt = "" if ev["attempt_no"] is None else f"a{ev['attempt_no']}"
        lines.append(f"  {ev['seq']:>2}  {ev['stage']:<14} {attempt:<3} {ev['outcome']:<8} "
                     f"{ev['duration_ms']:>9.1f} ms")
    return lines


def run_demo(conf: PipelineConfig, base: Path, enroll_image: Path, verify_image: Path,
             out: Callable[[str], None] = print) -> VerificationOutcome:
    """Enroll `enroll_image`, verify `verify_image` against it, in a store under `base`."""
    from fpv.catalog import Catalog, parse_label

    label = parse_label(enroll_image)
    if label is None:
        raise ValueError(f"{enroll_image} is not a labeled SOCOFing sample")
    identity = label.identity
    catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    with Repository(base / "db" / "fpv.sqlite", base / "templates", base / "probes", artifact_root=base) as repo:
        repo.migrate()
        ctl = PipelineController(conf, repo, catalog)
        session = str(uuid.uuid4())
        enrolled = ctl.enroll(enroll_image, identity, session_id=session)
        out(f"enroll {enroll_image.name} as {identity}: {enrolled.status.value} - {enrolled.message}")
        for line in format_trail(repo.stage_events(enrolled.run_id)):
            out(line)
        if enrolled.status is not Status.ENROLLED:
            return VerificationOutcome(enrolled.run_id, enrolled.status, enrolled.reason, enrolled.message,
                                       enrolled.quality_before, enrolled.attempts, None, None,
                                       conf.thresholds.bozorth3_match, None)
        verified = ctl.verify(verify_image, identity, session_id=session)
        out(f"verify {verify_image.name} as {identity}: score={verified.score} "
            f"threshold={verified.threshold}")
        for line in format_trail(repo.stage_events(verified.run_id)):
            out(line)
        out(verified.status.value)
        return verified


def main(argv: Sequence[str] | None = None) -> int:
    from fpv import config as cfgmod

    ap = argparse.ArgumentParser(prog="python -m fpv.pipeline", description="fpv pipeline controller")
    ap.add_argument("--demo", action="store_true",
                    help="enroll a Real sample and verify its Altered-Easy CR variant in a temporary store")
    ap.add_argument("--config", default=None, help="configuration file (default: frozen or pipeline.v1.yaml)")
    ap.add_argument("--enroll-image", type=Path, default=None,
                    help="default: <raw>/Real/1__M_Left_index_finger.BMP")
    ap.add_argument("--verify-image", type=Path, default=None,
                    help="default: <raw>/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP")
    ap.add_argument("--dir", type=Path, default=None, help="store directory (default: a new temporary directory)")
    args = ap.parse_args(argv)
    if not args.demo:
        ap.print_help()
        return 2
    conf = cfgmod.load(args.config)
    raw = conf.path("raw")
    enroll_image = args.enroll_image or raw / "Real" / "1__M_Left_index_finger.BMP"
    verify_image = args.verify_image or raw / "Altered" / "Altered-Easy" / "1__M_Left_index_finger_CR.BMP"
    base = args.dir or Path(tempfile.mkdtemp(prefix="fpv-pipeline-demo-"))
    base.mkdir(parents=True, exist_ok=True)
    print(f"config {conf.source_path} ({conf.config_set_id}); store {base}")
    outcome = run_demo(conf, base, enroll_image, verify_image)
    return 0 if outcome.status is Status.MATCH else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

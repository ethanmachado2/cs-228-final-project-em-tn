"""Threshold calibration, configuration freeze and the test-split guard (Req 14).

Calibration (Req 14.1-14.4) reads the `scores.csv` / `enrollments.csv` written by `fpv evaluate`
for the **train and validation splits only** of the provisional configuration:
`<reports>/eval/<split>/<config_set_id>/`. Inputs are filtered by path: every file must sit at
`eval/<split>/<cs>/<name>.csv` with `<split>` in `CALIBRATION_SPLITS`, so test-split scores can
never be read here (`check_calibration_input`).

- `threshold_at(genuine, impostor, target)` is `min{t in Z>=0 : FMR(t) <= target}` over completed
  impostor comparisons, with FNMR at that threshold and a low-confidence flag when there are
  fewer than `10 / target` impostor comparisons.
- `sweep` replays the recorded acquisition data (raw NFIQ 2 `quality_before` and the adapted
  minutiae count of every attempt that ran) under alternative `nfiq2_min` / `minutiae_min`
  values and reports FTE and FTA. Raising `minutiae_min` above the evaluated value can need an
  attempt that never ran, and lowering `nfiq2_min` below it admits samples that were never
  processed; those samples are counted as "unknown" and the rates are reported as a
  [low, high] range (unknown counted as acquired / as rejected). Everything else is exact.
- `freeze` writes `configs/pipeline.v1.frozen.yaml`: the provisional config with
  `frozen: true`, the chosen thresholds, the operating points, and `provenance` (source runs
  with SHA-256 digests, date, `pipeline_version`); `config.load` recomputes its
  `config_set_id`, which differs from the provisional one.

The test-split guard (Req 14.6, 14.7): `fpv evaluate --split test` requires a frozen config and
refuses to run when `<reports>/test_report_<config_set_id>.json` exists unless `--force` is
given; a forced run records the override (and the replaced report) in the new report. On
completion the CLI writes that report and `<reports>/baseline_v1.md`.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from fpv import config as cfgmod
from fpv.evaluate import (COMPLETED, DIFFICULTIES, REJECTED, EnrollResult, TrialResult, _metrics_table, _pct,
                          breakdowns, compute_metrics, fte)
from fpv.types import Status

if TYPE_CHECKING:
    from fpv.evaluate import EvaluationReport

CALIBRATION_SPLITS = ("train", "validation")
NFIQ2_CUTOFFS = tuple(range(0, 31, 5))
MINUTIAE_MINS = tuple(range(10, 21))
# Acquisition failures that are tool errors, not rejections (fixed across every sweep cell).
TOOL_ERROR_REASONS = frozenset({"nfiq2_tool_error", "preprocess_error"})
BASELINE_NAME = "baseline_v1.md"


class CalibrationInputError(ValueError):
    """A calibration input is missing or is not a train/validation evaluation output."""


class TestSplitGuardError(Exception):
    """`fpv evaluate --split test` is not allowed for this configuration (Req 14.6)."""

    __test__ = False  # not a pytest test class


# --------------------------------------------------------------------------- inputs


def check_calibration_input(path: Path) -> str:
    """The split of an evaluation output file; raises unless it is a train/validation output.

    Expected layout: `.../eval/<split>/<config_set_id>/<file>.csv` (Req 14.1).
    """
    path = Path(path)
    split = path.parent.parent.name
    if path.parent.parent.parent.name != "eval" or split not in CALIBRATION_SPLITS:
        raise CalibrationInputError(
            f"{path}: calibration reads only eval/{{{','.join(CALIBRATION_SPLITS)}}}/<config_set_id>/ outputs")
    return split


def filter_calibration_inputs(paths: Iterable[Path]) -> list[Path]:
    """Keep only train/validation evaluation outputs (drops test-split and foreign paths)."""
    kept = []
    for p in paths:
        try:
            check_calibration_input(p)
        except CalibrationInputError:
            continue
        kept.append(Path(p))
    return kept


def calibration_inputs(conf: cfgmod.PipelineConfig) -> dict[str, Path]:
    """`{split: eval dir}` of the train and validation runs of `conf` (both must exist)."""
    root = conf.path("reports") / "eval"
    candidates = sorted(root.glob(f"*/{conf.config_set_id}/scores.csv"))
    found = {check_calibration_input(p): p.parent for p in filter_calibration_inputs(candidates)}
    missing = [s for s in CALIBRATION_SPLITS if s not in found]
    if missing:
        raise CalibrationInputError(
            f"no evaluation of split(s) {', '.join(missing)} for {conf.config_set_id} under {root}; run "
            + " and ".join(f"`fpv --config {conf.source_path} evaluate --split {s}`" for s in missing))
    return {s: found[s] for s in CALIBRATION_SPLITS}


def _opt_int(text: str) -> int | None:
    return None if text in ("", None) else int(text)


def _ints(text: str) -> tuple[int | None, ...]:
    return () if not text else tuple(_opt_int(v) for v in text.split(";"))


def load_trials(path: Path) -> list[TrialResult]:
    check_calibration_input(path)
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(TrialResult(
                trial_id=r["trial_id"], run_id=r["run_id"], probe_sample=r["probe_sample"],
                probe_subject=int(r["probe_subject"]), claim_user=int(r["claim_user"]),
                claim_hand=r["claim_hand"], claim_finger=r["claim_finger"], genuine=r["genuine"] == "True",
                alteration=r["alteration"], difficulty=r["difficulty"], status=r["status"],
                reason=r["reason"] or None, score=_opt_int(r["score"]), threshold=int(r["threshold"]),
                quality_before=_opt_int(r["quality_before"]), attempts_used=int(r["attempts_used"]),
                minutiae_counts=r["minutiae_counts"], quality_after=r["quality_after"],
                cache_hit=r["cache_hit"] == "True"))
    return out


def load_enrollments(path: Path) -> list[EnrollResult]:
    check_calibration_input(path)
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(EnrollResult(
                enroll_id=r["enroll_id"], run_id=r["run_id"], sample=r["sample"], user_id=int(r["user_id"]),
                hand=r["hand"], finger=r["finger"], status=r["status"], reason=r["reason"] or None,
                quality_before=_opt_int(r["quality_before"]), attempts_used=int(r["attempts_used"]),
                minutiae_counts=r["minutiae_counts"], quality_after=r["quality_after"],
                template_id=r["template_id"] or None))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class SplitScores:
    split: str
    eval_dir: Path
    trials: list[TrialResult]
    enrollments: list[EnrollResult]
    digests: dict[str, str]  # file name -> sha256


def load_split(eval_dir: Path) -> SplitScores:
    scores, enrolls = Path(eval_dir) / "scores.csv", Path(eval_dir) / "enrollments.csv"
    split = check_calibration_input(scores)
    return SplitScores(split, Path(eval_dir), load_trials(scores), load_enrollments(enrolls),
                       {"scores.csv": _sha256(scores), "enrollments.csv": _sha256(enrolls)})


def completed_scores(trials: Iterable[TrialResult]) -> tuple[list[int], list[int]]:
    """(genuine, impostor) scores of completed comparisons (MATCH / NO_MATCH)."""
    gen, imp = [], []
    for t in trials:
        if t.status in COMPLETED and t.score is not None:
            (gen if t.genuine else imp).append(t.score)
    return gen, imp


# --------------------------------------------------------------------------- operating points


def min_impostors(target: float) -> int:
    """Impostor comparisons needed for a confident FMR estimate at `target` (10 / target)."""
    return math.ceil(10.0 / target - 1e-9)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    target_fmr: float
    threshold: int
    fmr: float
    fnmr: float
    impostor: int
    genuine: int
    false_matches: int
    false_non_matches: int
    min_impostors: int
    low_confidence: bool


def threshold_at(genuine: Sequence[int], impostor: Sequence[int], target: float) -> OperatingPoint:
    """Smallest integer threshold `t >= 0` with FMR(t) <= target, and FNMR at it (Req 14.2, 14.3).

    FMR(t) = |impostor >= t| / |impostor| is non-increasing in `t` and 0 at max + 1, so the
    minimum always exists.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target FMR must be in (0, 1), got {target}")
    if len(impostor) == 0:
        raise ValueError("no completed impostor comparisons")
    imp = np.sort(np.asarray(impostor, dtype=np.int64))
    n = len(imp)
    ts = np.arange(0, int(imp[-1]) + 2)
    false_matches = n - np.searchsorted(imp, ts, side="left")
    # Integer comparison (count <= target * n) avoids float noise on the boundary.
    ok = false_matches <= math.floor(target * n + 1e-9)
    t = int(ts[int(np.argmax(ok))])
    fm = int(false_matches[t])
    gen = np.asarray(genuine, dtype=np.int64)
    fnm = int((gen < t).sum())
    need = min_impostors(target)
    return OperatingPoint(target, t, fm / n, (fnm / len(gen)) if len(gen) else math.nan, n, len(gen), fm, fnm,
                          need, n < need)


# --------------------------------------------------------------------------- FTE / FTA sweeps


@dataclass(frozen=True, slots=True)
class Acquisition:
    """One acquisition (an enrollment sample, or one probe) as recorded by the evaluation."""

    status: str
    reason: str | None
    quality_before: int | None
    counts: tuple[int | None, ...]  # adapted minutiae count per attempt that ran


def acquisitions(split: SplitScores) -> tuple[list[Acquisition], list[Acquisition]]:
    """(enrollment acquisitions, unique probe acquisitions) of one split."""
    enr = [Acquisition(e.status, e.reason, e.quality_before, _ints(e.minutiae_counts)) for e in split.enrollments]
    probes: dict[str, Acquisition] = {}
    for t in split.trials:
        a = Acquisition(t.status, t.reason, t.quality_before, _ints(t.minutiae_counts))
        prev = probes.get(t.probe_sample)
        # The probe's acquisition is shared by its trials (probe cache); a rejection wins.
        if prev is None or (t.status in REJECTED and prev.status not in REJECTED):
            probes[t.probe_sample] = a
    return enr, [probes[k] for k in sorted(probes)]


def acquisition_outcome(a: Acquisition, nfiq2_min: int, minutiae_min: int, *, evaluated_nfiq2_min: int,
                        max_attempts: int) -> str:
    """'ok', 'rejected', 'unknown' or 'error' under alternative gate values.

    Replays the attempt policy on the recorded per-attempt counts: the first attempt reaching
    `minutiae_min` is accepted. The recorded attempts are a prefix of the policy's attempts, so
    the outcome is exact unless it needs an attempt that did not run, or the sample never
    reached the attempt loop (raw score below the evaluated `nfiq2_min`).
    """
    if a.reason in TOOL_ERROR_REASONS:
        return "error"
    q = a.quality_before
    if q is None:  # unreadable or NFIQ 2 could not score it: rejected at any cutoff
        return "rejected"
    if q < nfiq2_min:
        return "rejected"
    if q < evaluated_nfiq2_min or not a.counts:
        return "unknown"
    counts = [c or 0 for c in a.counts]
    if any(c >= minutiae_min for c in counts):
        return "ok"
    return "rejected" if len(counts) >= max_attempts else "unknown"


@dataclass(frozen=True, slots=True)
class SweepCell:
    nfiq2_min: int
    minutiae_min: int
    kind: str  # "FTE" | "FTA"
    n: int
    rejected: int
    unknown: int
    errors: int
    low: float   # unknown counted as acquired
    high: float  # unknown counted as rejected

    @property
    def exact(self) -> bool:
        return self.unknown == 0


def sweep_cell(acqs: Sequence[Acquisition], kind: str, nfiq2_min: int, minutiae_min: int, *,
               evaluated_nfiq2_min: int, max_attempts: int) -> SweepCell:
    """FTE counts tool errors as failures (as `evaluate.fte` does); FTA does not."""
    outcomes = [acquisition_outcome(a, nfiq2_min, minutiae_min, evaluated_nfiq2_min=evaluated_nfiq2_min,
                                    max_attempts=max_attempts) for a in acqs]
    n = len(outcomes)
    rej, unk, err = outcomes.count("rejected"), outcomes.count("unknown"), outcomes.count("error")
    base = rej + (err if kind == "FTE" else 0)
    return SweepCell(nfiq2_min, minutiae_min, kind, n, rej, unk, err,
                     base / n if n else math.nan, (base + unk) / n if n else math.nan)


def sweep(enr: Sequence[Acquisition], probes: Sequence[Acquisition], *, evaluated_nfiq2_min: int,
          max_attempts: int, nfiq2_cutoffs: Sequence[int] = NFIQ2_CUTOFFS,
          minutiae_mins: Sequence[int] = MINUTIAE_MINS) -> list[SweepCell]:
    """FTE and FTA over the full (nfiq2_min x minutiae_min) grid (Req 14.4)."""
    cells = []
    for q in nfiq2_cutoffs:
        for m in minutiae_mins:
            for kind, acqs in (("FTE", enr), ("FTA", probes)):
                cells.append(sweep_cell(acqs, kind, q, m, evaluated_nfiq2_min=evaluated_nfiq2_min,
                                        max_attempts=max_attempts))
    return cells


def quality_cutoff_accuracy(splits: Sequence[SplitScores], cutoff: int, threshold: int) -> dict[str, Any]:
    """FMR / FNMR at `threshold` over the comparisons that a stricter raw-quality gate keeps.

    Exact for cutoffs at or above the evaluated `nfiq2_min`: a comparison survives when both the
    probe and the claimed finger's enrollment sample score at least `cutoff`; scores are unchanged.
    """
    kept: list[TrialResult] = []
    for s in splits:
        enrolled_q = {(e.user_id, e.hand, e.finger): e.quality_before for e in s.enrollments}
        for t in s.trials:
            eq = enrolled_q.get((t.claim_user, t.claim_hand, t.claim_finger))
            if t.quality_before is not None and t.quality_before >= cutoff and eq is not None and eq >= cutoff:
                kept.append(t)
    gen, imp = completed_scores(kept)
    return {"nfiq2_min": cutoff, "genuine": len(gen), "impostor": len(imp),
            "fnmr": (sum(s < threshold for s in gen) / len(gen)) if gen else math.nan,
            "fmr": (sum(s >= threshold for s in imp) / len(imp)) if imp else math.nan}


# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    source_config: Path
    source_config_set_id: str
    splits: tuple[SplitScores, ...]
    targets: tuple[float, ...]
    primary_target_fmr: float
    pooled: tuple[OperatingPoint, ...]
    per_split: dict[str, tuple[OperatingPoint, ...]]
    metrics_at_primary: dict[str, Any]  # split -> Metrics (and "pooled")
    fte_at_primary: dict[str, float]
    sweep: list[SweepCell]
    quality_cutoffs: list[dict[str, Any]]
    evaluated: dict[str, int]  # nfiq2_min, minutiae_min, bozorth3_match, max_attempts
    score_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> OperatingPoint:
        return next(op for op in self.pooled if op.target_fmr == self.primary_target_fmr)


def _calibration_section(conf: cfgmod.PipelineConfig) -> tuple[tuple[float, ...], float]:
    cal = conf.raw.get("calibration") or {}
    targets = tuple(float(t) for t in cal.get("target_fmrs", (0.001, 0.01)))
    primary = float(cal.get("primary_target_fmr", 0.01))
    if not targets or any(not 0.0 < t < 1.0 for t in targets):
        raise CalibrationInputError(f"calibration.target_fmrs must be in (0, 1), got {list(targets)}")
    return targets, primary


def calibrate(conf: cfgmod.PipelineConfig, *, primary_target_fmr: float | None = None) -> CalibrationResult:
    """Operating points, per-split checks and FTE/FTA sweeps from train + validation (Req 14.1-14.4)."""
    if conf.frozen:
        raise CalibrationInputError(
            f"{conf.source_path} is already frozen; calibrate the provisional configuration "
            f"(`fpv --config {cfgmod.DEFAULT_CONFIG} calibrate`)")
    targets, primary = _calibration_section(conf)
    if primary_target_fmr is not None:
        primary = float(primary_target_fmr)
    if primary not in targets:
        raise CalibrationInputError(f"primary target FMR {primary} is not one of {list(targets)}")
    splits = tuple(load_split(d) for d in calibration_inputs(conf).values())

    all_trials = [t for s in splits for t in s.trials]
    gen, imp = completed_scores(all_trials)
    pooled = tuple(threshold_at(gen, imp, t) for t in targets)
    per_split = {s.split: tuple(threshold_at(*completed_scores(s.trials), t) for t in targets) for s in splits}
    primary_t = next(op for op in pooled if op.target_fmr == primary).threshold

    metrics = {s.split: compute_metrics(s.trials, primary_t) for s in splits}
    metrics["pooled"] = compute_metrics(all_trials, primary_t)
    fte_values = {s.split: fte(e.status for e in s.enrollments) for s in splits}
    fte_values["pooled"] = fte(e.status for s in splits for e in s.enrollments)

    evaluated = {"nfiq2_min": conf.thresholds.nfiq2_min, "minutiae_min": conf.thresholds.minutiae_min,
                 "bozorth3_match": conf.thresholds.bozorth3_match, "max_attempts": conf.max_attempts}
    enr: list[Acquisition] = []
    probes: list[Acquisition] = []
    for s in splits:
        e, p = acquisitions(s)
        enr += e
        probes += p
    cells = sweep(enr, probes, evaluated_nfiq2_min=conf.thresholds.nfiq2_min, max_attempts=conf.max_attempts)
    cutoffs = [quality_cutoff_accuracy(splits, c, primary_t) for c in NFIQ2_CUTOFFS
               if c >= conf.thresholds.nfiq2_min]
    stats = {
        "genuine": len(gen), "impostor": len(imp),
        "genuine_median": float(np.median(gen)) if gen else math.nan,
        "impostor_median": float(np.median(imp)) if imp else math.nan,
        "impostor_p99": float(np.percentile(imp, 99)) if imp else math.nan,
        "impostor_p999": float(np.percentile(imp, 99.9)) if imp else math.nan,
        "impostor_max": int(max(imp)) if imp else None,
    }
    return CalibrationResult(conf.source_path, conf.config_set_id, splits, targets, primary, pooled, per_split,
                             metrics, fte_values, cells, cutoffs, evaluated, stats)


# --------------------------------------------------------------------------- reports


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _rng(cell: SweepCell) -> str:
    return _pct(cell.low) if cell.exact else f"{_pct(cell.low)}–{_pct(cell.high)}"


def _cell(cells: Sequence[SweepCell], kind: str, q: int, m: int) -> SweepCell:
    return next(c for c in cells if c.kind == kind and c.nfiq2_min == q and c.minutiae_min == m)


def calibration_dir(conf: cfgmod.PipelineConfig) -> Path:
    return conf.path("reports") / "calibration" / conf.config_set_id


def render_calibration(res: CalibrationResult, conf: cfgmod.PipelineConfig, *, frozen_path: Path | None,
                       frozen_id: str | None, decisions: Mapping[str, Any]) -> str:
    ev = res.evaluated
    q0, m0 = ev["nfiq2_min"], ev["minutiae_min"]
    lines = [
        f"# Calibration of `{res.source_config_set_id}`",
        "",
        f"- source config: `{_rel(res.source_config, conf.root)}` (config_set_id `{res.source_config_set_id}`); "
        f"evaluated with nfiq2_min {q0}, minutiae_min {m0}, bozorth3_match {ev['bozorth3_match']}",
        "- inputs (train and validation only, Req 14.1): "
        + "; ".join(f"`{_rel(s.eval_dir / 'scores.csv', conf.root)}` ({len(s.trials)} trials, "
                    f"{len(s.enrollments)} enrollments, sha256 {s.digests['scores.csv'][:12]})" for s in res.splits),
        f"- completed comparisons: {res.score_stats['genuine']} genuine, {res.score_stats['impostor']} impostor; "
        f"score medians genuine {res.score_stats['genuine_median']:g}, impostor "
        f"{res.score_stats['impostor_median']:g}; impostor p99 {res.score_stats['impostor_p99']:g}, "
        f"p99.9 {res.score_stats['impostor_p999']:g}, max {res.score_stats['impostor_max']}",
        "",
        "## Operating points (Req 14.2, 14.3)",
        "",
        "Threshold = smallest integer Bozorth3 score with empirical FMR at or below the target, over "
        "completed impostor comparisons. Low confidence when impostors < 10 / target.",
        "",
        "| data | target FMR | threshold | FMR | false matches | FNMR | false non-matches | impostors "
        "(needed) | confidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = [("train + validation", op) for op in res.pooled]
    rows += [(split, op) for split, ops in res.per_split.items() for op in ops]
    for name, op in rows:
        primary = " (primary)" if name == "train + validation" and op.target_fmr == res.primary_target_fmr else ""
        lines.append(f"| {name}{primary} | {op.target_fmr:g} | {op.threshold} | {_pct(op.fmr)} | {op.false_matches} | "
                     f"{_pct(op.fnmr)} | {op.false_non_matches} | {op.impostor} ({op.min_impostors}) | "
                     f"{'LOW' if op.low_confidence else 'ok'} |")
    t = res.primary.threshold
    lines += ["", f"## Rates at the primary threshold {t}", "",
              "| data | FTE | FTA | FMR | FNMR | gFMR | gFNMR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in res.metrics_at_primary.items():
        lines.append(f"| {name} | {_pct(res.fte_at_primary[name])} | {_pct(m.fta)} | {_pct(m.fmr)} | {_pct(m.fnmr)} | "
                     f"{_pct(m.gfmr)} | {_pct(m.gfnmr)} |")
    pooled_trials = [x for s in res.splits for x in s.trials]
    for name, brk in breakdowns(pooled_trials, t).items():
        if name == "difficulty x alteration":
            continue
        lines += ["", f"### Train + validation by {name} at threshold {t}", "", *_metrics_table(brk, name)]

    lines += [
        "", "## FTE / FTA sweeps (Req 14.4)", "",
        "Replayed offline from the recorded raw NFIQ 2 score and per-attempt adapted minutiae counts "
        "(train + validation pooled). A range `low–high` means some samples would need an attempt that did "
        "not run (minutiae_min above the evaluated value) or were never processed (nfiq2_min below it); "
        "low counts them as acquired, high as rejected. Single values are exact.", "",
        f"### NFIQ 2 cutoff (minutiae_min {m0})", "",
        f"| nfiq2_min | FTE | FTA | FNMR at {t} | FMR at {t} | genuine kept | impostor kept |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    acc = {c["nfiq2_min"]: c for c in res.quality_cutoffs}
    for q in NFIQ2_CUTOFFS:
        a = acc.get(q)
        tail = (f"{_pct(a['fnmr'])} | {_pct(a['fmr'])} | {a['genuine']} | {a['impostor']}" if a
                else "n/a | n/a | n/a | n/a")
        mark = " (evaluated)" if q == q0 else ""
        lines.append(f"| {q}{mark} | {_rng(_cell(res.sweep, 'FTE', q, m0))} | "
                     f"{_rng(_cell(res.sweep, 'FTA', q, m0))} | {tail} |")
    lines += ["", f"FNMR/FMR columns: comparisons kept when both the probe and the claimed finger's enrollment "
                  f"sample score at least the cutoff (exact for cutoffs >= {q0}).", "",
              f"### minutiae_min (nfiq2_min {q0})", "", "| minutiae_min | FTE | FTA |", "|---:|---:|---:|"]
    for m in MINUTIAE_MINS:
        mark = " (evaluated)" if m == m0 else ""
        lines.append(f"| {m}{mark} | {_rng(_cell(res.sweep, 'FTE', q0, m))} | {_rng(_cell(res.sweep, 'FTA', q0, m))} |")
    lines += ["", "The full nfiq2_min x minutiae_min grid is in `calibration.json`.", "", "## Decisions", ""]
    for k, v in decisions.items():
        lines.append(f"- {k}: {v}")
    if frozen_path is not None:
        lines += ["", f"Frozen config: `{_rel(frozen_path, conf.root)}` (config_set_id `{frozen_id}`)."]
    lines.append("")
    return "\n".join(lines)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    return obj


def calibration_json(res: CalibrationResult, conf: cfgmod.PipelineConfig, decisions: Mapping[str, Any],
                     frozen_id: str | None) -> dict[str, Any]:
    return _jsonable({
        "source_config": _rel(res.source_config, conf.root), "source_config_set_id": res.source_config_set_id,
        "frozen_config_set_id": frozen_id,
        "inputs": [{"split": s.split, "dir": _rel(s.eval_dir, conf.root), "trials": len(s.trials),
                    "enrollments": len(s.enrollments), "sha256": s.digests} for s in res.splits],
        "evaluated": res.evaluated, "score_stats": res.score_stats, "targets": res.targets,
        "primary_target_fmr": res.primary_target_fmr, "operating_points": res.pooled,
        "operating_points_per_split": res.per_split,
        "at_primary": {k: asdict(v) for k, v in res.metrics_at_primary.items()},
        "fte_at_primary": res.fte_at_primary, "sweep": res.sweep, "quality_cutoffs": res.quality_cutoffs,
        "decisions": decisions,
    })


# --------------------------------------------------------------------------- freeze


def freeze(conf: cfgmod.PipelineConfig, res: CalibrationResult, out_path: Path, *,
           nfiq2_min: int | None = None, minutiae_min: int | None = None,
           report_path: Path | None = None, pipeline_version: str | None = None,
           decisions: Mapping[str, Any] | None = None, now: datetime | None = None,
           force: bool = False) -> cfgmod.PipelineConfig:
    """Write the frozen configuration (Req 14.5) and return it loaded and validated."""
    if conf.frozen:
        raise CalibrationInputError(f"{conf.source_path} is already frozen")
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise CalibrationInputError(f"{out_path} already exists; pass --force to replace it")
    raw = copy.deepcopy(dict(conf.raw))
    raw["frozen"] = True
    th = dict(raw["thresholds"])
    th["bozorth3_match"] = res.primary.threshold
    if nfiq2_min is not None:
        th["nfiq2_min"] = int(nfiq2_min)
    if minutiae_min is not None:
        th["minutiae_min"] = int(minutiae_min)
    raw["thresholds"] = th
    raw["calibration"] = {
        "target_fmrs": list(res.targets),
        "primary_target_fmr": res.primary_target_fmr,
        "operating_points": [{"target_fmr": op.target_fmr, "bozorth3_match": op.threshold,
                              "low_confidence": op.low_confidence} for op in res.pooled],
    }
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    raw["provenance"] = _jsonable({
        "calibrated_at": stamp,
        "pipeline_version": pipeline_version or cfgmod.pipeline_version(),
        "source_config": _rel(conf.source_path, conf.root),
        "source_config_set_id": conf.config_set_id,
        "source_runs": [{"split": s.split, "dir": _rel(s.eval_dir, conf.root), "trials": len(s.trials),
                         "enrollments": len(s.enrollments), "sha256": s.digests} for s in res.splits],
        "calibration_report": _rel(report_path, conf.root) if report_path else None,
        "operating_points": [{"target_fmr": op.target_fmr, "bozorth3_match": op.threshold, "fmr": op.fmr,
                              "fnmr": op.fnmr, "impostor": op.impostor, "genuine": op.genuine}
                             for op in res.pooled],
        "decisions": dict(decisions or {}),
    })
    header = (
        "# fpv pipeline configuration, schema v1 (FROZEN by `fpv calibrate`; do not edit by hand).\n"
        f"# Calibrated from {conf.config_set_id} train + validation scores; see provenance.\n"
        "# Everything except `provenance` is hashed into config_set_id. `fpv evaluate --split test` runs once\n"
        "# per config_set_id (test_report_<config_set_id>.json).\n"
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
    frozen = cfgmod.load(tmp, root=conf.root)
    if not frozen.frozen or frozen.config_set_id == conf.config_set_id:
        tmp.unlink()
        raise RuntimeError("frozen config did not get a new config_set_id")
    tmp.replace(out_path)
    return cfgmod.load(out_path, root=conf.root)


def run_calibration(conf: cfgmod.PipelineConfig, *, out_config: Path | None = None,
                    primary_target_fmr: float | None = None, nfiq2_min: int | None = None,
                    minutiae_min: int | None = None, force: bool = False,
                    pipeline_version: str | None = None) -> tuple[CalibrationResult, cfgmod.PipelineConfig, Path]:
    """`fpv calibrate`: calibrate, write the report and freeze. Returns (result, frozen config, report)."""
    res = calibrate(conf, primary_target_fmr=primary_target_fmr)
    out_config = Path(out_config) if out_config else conf.root / cfgmod.FROZEN_CONFIG
    if out_config.exists() and not force:
        raise CalibrationInputError(f"{out_config} already exists; pass --force to replace it")
    out_dir = calibration_dir(conf)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "calibration.md"
    decisions = {
        "primary operating point": f"FMR {res.primary_target_fmr:g} -> bozorth3_match {res.primary.threshold}",
        "nfiq2_min": (f"{conf.thresholds.nfiq2_min} -> {nfiq2_min}" if nfiq2_min is not None
                      and nfiq2_min != conf.thresholds.nfiq2_min else f"{conf.thresholds.nfiq2_min} (unchanged)"),
        "minutiae_min": (f"{conf.thresholds.minutiae_min} -> {minutiae_min}" if minutiae_min is not None
                         and minutiae_min != conf.thresholds.minutiae_min
                         else f"{conf.thresholds.minutiae_min} (unchanged)"),
    }
    frozen = freeze(conf, res, out_config, nfiq2_min=nfiq2_min, minutiae_min=minutiae_min, report_path=report,
                    pipeline_version=pipeline_version, decisions=decisions, force=force)
    report.write_text(render_calibration(res, conf, frozen_path=out_config, frozen_id=frozen.config_set_id,
                                         decisions=decisions), encoding="utf-8")
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration_json(res, conf, decisions, frozen.config_set_id), indent=2) + "\n", encoding="utf-8")
    return res, frozen, report


# --------------------------------------------------------------------------- test-split guard / baseline


def test_report_path(conf: cfgmod.PipelineConfig) -> Path:
    return conf.path("reports") / f"test_report_{conf.config_set_id}.json"


test_report_path.__test__ = False  # type: ignore[attr-defined]


def guard_test_split(conf: cfgmod.PipelineConfig, *, force: bool) -> dict[str, Any] | None:
    """Raise unless the test split may be evaluated (Req 14.6); returns the report a --force run replaces."""
    if not conf.frozen:
        raise TestSplitGuardError("the test split requires a frozen configuration (run `fpv calibrate` first); "
                                  f"{conf.source_path} has frozen: false")
    path = test_report_path(conf)
    if not path.exists():
        return None
    if not force:
        raise TestSplitGuardError(f"a test report already exists for {conf.config_set_id} ({path}); the test split "
                                  "is evaluated once per frozen configuration. Pass --force to override "
                                  "(the override is recorded in the report)")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"unreadable": str(path)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a proportion k / n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def operating_thresholds(conf: cfgmod.PipelineConfig) -> list[tuple[float, int]]:
    cal = conf.raw.get("calibration") or {}
    ops = [(float(o["target_fmr"]), int(o["bozorth3_match"])) for o in cal.get("operating_points", [])]
    return ops or [(float(cal.get("primary_target_fmr", 0.01)), conf.thresholds.bozorth3_match)]


def write_test_report(conf: cfgmod.PipelineConfig, report: EvaluationReport, *, forced: bool,
                      previous: Mapping[str, Any] | None, pipeline_version: str,
                      now: datetime | None = None) -> Path:
    """`test_report_<config_set_id>.json`, recording any --force override (Req 14.6)."""
    gen, imp = completed_scores(report.trials)
    history = list((previous or {}).get("history", []))
    if previous:
        history.append({k: previous.get(k) for k in ("completed_at", "pipeline_version", "forced", "overall")})
    data = _jsonable({
        "config_set_id": conf.config_set_id,
        "config": _rel(conf.source_path, conf.root),
        "split": "test",
        "completed_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "pipeline_version": pipeline_version,
        "eval_dir": _rel(report.out_dir, conf.root),
        "forced": forced,
        "override": ({"forced": True, "replaced_report_completed_at": (previous or {}).get("completed_at")}
                     if forced else None),
        "history": history,
        "thresholds": asdict(conf.thresholds),
        "fte": report.fte,
        "overall": asdict(report.overall),
        "operating_points": [{"target_fmr": target, "bozorth3_match": t,
                              "fmr": compute_metrics(report.trials, t).fmr,
                              "fnmr": compute_metrics(report.trials, t).fnmr} for target, t in
                             operating_thresholds(conf)],
        "genuine": len(gen), "impostor": len(imp),
    })
    path = test_report_path(conf)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def render_baseline(conf: cfgmod.PipelineConfig, report: EvaluationReport, *, pipeline_version: str,
                    forced: bool, report_path: Path) -> str:
    """`baseline_v1.md`: test-split FTE, FTA, FMR and FNMR by alteration and difficulty (Req 14.7)."""
    trials, enrolls = report.trials, report.enrollments
    t = conf.thresholds.bozorth3_match
    m = report.overall
    gen, imp = completed_scores(trials)
    enrolled = sum(1 for e in enrolls if e.status == Status.ENROLLED.value)
    fm, fnm = sum(s >= t for s in imp), sum(s < t for s in gen)
    fm_ci, fnm_ci = wilson(fm, len(imp)), wilson(fnm, len(gen))
    prov = conf.raw.get("provenance") or {}
    cal = conf.raw.get("calibration") or {}

    def ci(lo_hi: tuple[float, float]) -> str:
        return f"[{_pct(lo_hi[0])}, {_pct(lo_hi[1])}]"

    lines = [
        "# Phase 1 baseline (v1): test split",
        "",
        f"- config: `{_rel(conf.source_path, conf.root)}`, config_set_id `{conf.config_set_id}` (frozen: "
        f"{conf.frozen}), calibrated from `{prov.get('source_config_set_id', 'n/a')}` train + validation on "
        f"{prov.get('calibrated_at', 'n/a')}",
        f"- thresholds: bozorth3_match {t} (primary operating point, target FMR "
        f"{cal.get('primary_target_fmr', 'n/a')}), nfiq2_min {conf.thresholds.nfiq2_min}, minutiae_min "
        f"{conf.thresholds.minutiae_min}",
        f"- pipeline_version: `{pipeline_version}`; seed {report.plan.seed}, impostors per probe {report.plan.k}, "
        f"{len(report.plan.subjects)} subjects",
        f"- evaluation outputs: `{_rel(report.out_dir, conf.root)}/` (scores.csv, enrollments.csv, summary.md, "
        f"det.png); guard record `{_rel(report_path, conf.root)}`"
        + ("; **run forced with --force (override recorded in the report)**" if forced else ""),
        "",
        "## Overall",
        "",
        "| metric | value | 95% CI | counts |",
        "|---|---:|---|---|",
        f"| FTE | {_pct(report.fte)} | | {len(enrolls) - enrolled} of {len(enrolls)} Real enrollment samples |",
        f"| FTA | {_pct(m.fta)} | | {m.rejected_probes} of {m.probes} Altered probes rejected |",
        f"| FMR | {_pct(m.fmr)} | {ci(fm_ci)} | {fm} of {m.impostor} impostor comparisons |",
        f"| FNMR | {_pct(m.fnmr)} | {ci(fnm_ci)} | {fnm} of {m.genuine} genuine comparisons |",
        f"| generalized FMR | {_pct(m.gfmr)} | | (1 - FTA) x FMR |",
        f"| generalized FNMR | {_pct(m.gfnmr)} | | FTA + (1 - FTA) x FNMR |",
        "",
        "FTE is reported once: enrollments use Real samples only, which have no alteration type or "
        "difficulty. FTA, FMR and FNMR are broken down below. NOT_ENROLLED trials "
        f"({m.not_enrolled}, claimed finger failed to enroll) and ERROR trials ({m.errors}) are excluded "
        "from the rates.",
        "",
        "## Calibrated operating points on the test split",
        "",
        "| target FMR | threshold | test FMR | test FNMR | by difficulty: FNMR Easy / Medium / Hard |",
        "|---:|---:|---:|---:|---|",
    ]
    for target, thr in operating_thresholds(conf):
        mm = compute_metrics(trials, thr)
        per = " / ".join(_pct(compute_metrics([x for x in trials if x.difficulty == d], thr).fnmr)
                         for d in DIFFICULTIES)
        lines.append(f"| {target:g} | {thr} | {_pct(mm.fmr)} | {_pct(mm.fnmr)} | {per} |")
    for name, rows in breakdowns(trials, t).items():
        lines += ["", f"## By {name} (threshold {t})", "", *_metrics_table(rows, name)]
    if gen and imp:
        lines += ["", f"Score medians: genuine {float(np.median(gen)):g}, impostor {float(np.median(imp)):g}; "
                      f"impostor max {max(imp)}."]
    lines += ["", f"![DET]({_rel(report.out_dir, conf.path('reports'))}/det.png)", ""]
    return "\n".join(lines)


def write_baseline(conf: cfgmod.PipelineConfig, report: EvaluationReport, *, pipeline_version: str,
                   forced: bool, report_path: Path) -> Path:
    path = conf.path("reports") / BASELINE_NAME
    path.write_text(render_baseline(conf, report, pipeline_version=pipeline_version, forced=forced,
                                    report_path=report_path), encoding="utf-8")
    return path

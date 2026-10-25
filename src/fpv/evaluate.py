"""Batch evaluation harness: trial generation, parallel execution, metrics and the DET plot (Req 13).

Protocol for one split (Req 13.1-13.4, 13.9):

- every labeled, non-excluded Real sample of the (optionally seeded-subset) subjects is enrolled;
- every labeled, non-excluded Altered probe whose finger has a Real enrollment gets one genuine
  trial (claiming its own identity) and `k` impostor trials against other subjects' enrolled
  identities with the same hand and finger, drawn with `numpy.random.default_rng([seed, h])`
  over the sorted candidate list, where `h` is a stable hash of the probe key. Real samples are
  never probes.

`generate_trials` reads only the split/exclusion tables, so the trial list is a pure function of
`(catalog, split, seed, k, subjects)` (Req 13.5).

Execution (Req 13.5, 13.6): trials are grouped into one work unit per (hand, finger), which holds
that finger position's enrollments and every trial claiming it (impostors share hand and finger,
so units are independent). Each unit runs in a worker process through `PipelineController` with
its own store, `<out>/store/<Hand>_<finger>/` (SQLite DB `fpv.sqlite`, `templates/`, `probes/`).
Per-unit stores instead of one shared WAL database: the eval directory sits on a bind mount, where
SQLite's WAL shared-memory file is not shared with readers outside the container (a host reader
corrupted a live DB during the first demo run), and the unit -> store mapping does not depend on
the worker count. Each unit's DB is written on process-local temp storage, checkpointed, switched
to rollback-journal mode and only then moved into the store. Every trial is persisted as a run. A per-unit in-memory
probe cache keyed by `(sha256, config_set_id)` reuses a probe's processed template across its
trials; the controller records `cache_hit` in the `quality_before` stage details (true on a
replay, false on a miss). Results are sorted by `trial_id` before writing.

Outputs under `<reports>/eval/<split>/<config_set_id>/` (Req 13.7): `scores.csv` (one row per
trial), `enrollments.csv`, `summary.md` (FTE, FTA, FMR, FNMR and the generalized rates at the
active threshold, broken down by alteration type and difficulty) and `det.png`.

Metrics (Req 13.8, ISO/IEC 19795-1 style):
    fte   = 1 - enrolled / attempted_enrollments
    fta   = probes rejected for quality or minutiae / probes
    fmr   = |impostor scores >= t| / |completed impostor comparisons|
    fnmr  = |genuine scores < t| / |completed genuine comparisons|
    gfmr  = (1 - fta) * fmr
    gfnmr = fta + (1 - fta) * fnmr
Completed comparisons are trials ending MATCH or NO_MATCH. NOT_ENROLLED (the claimed finger
failed to enroll) and ERROR trials are excluded from every rate and counted separately.
"""

from __future__ import annotations

import csv
import hashlib
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np

from fpv.types import Identity, Status

if TYPE_CHECKING:
    from fpv.catalog import Catalog
    from fpv.config import PipelineConfig
    from fpv.pipeline import CachedProbe, Stages

ALTERATIONS = ("Obl", "CR", "Zcut")
DIFFICULTIES = ("Easy", "Medium", "Hard")
REJECTED = frozenset({Status.REJECTED_QUALITY.value, Status.REJECTED_MINUTIAE.value})
COMPLETED = frozenset({Status.MATCH.value, Status.NO_MATCH.value})


# --------------------------------------------------------------------------- trial generation


@dataclass(frozen=True, slots=True)
class Enrollment:
    enroll_id: str
    sample: str  # dataset key, relative to the raw root (e.g. "Real/1__M_Left_index_finger.BMP")
    identity: Identity


@dataclass(frozen=True, slots=True)
class Trial:
    trial_id: str
    probe: str  # dataset key of an Altered sample
    probe_identity: Identity
    claim: Identity
    genuine: bool
    alteration: str
    difficulty: str


@dataclass(frozen=True, slots=True)
class TrialPlan:
    split: str
    seed: int
    k: int
    subjects: tuple[int, ...]  # the subjects evaluated (all of the split's when not subset)
    enrollments: tuple[Enrollment, ...]
    trials: tuple[Trial, ...]
    skipped_probes: int  # Altered probes whose finger has no Real enrollment in the split


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16], 16)


def select_subjects(all_subjects: Iterable[int], n: int | None, seed: int) -> tuple[int, ...]:
    """Seeded subset of `n` subjects (Req 13.9); every subject when `n` is None or too large."""
    pool = sorted(set(all_subjects))
    if n is None or n >= len(pool):
        return tuple(pool)
    if n < 1:
        raise ValueError(f"--subjects must be at least 1, got {n}")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False)
    return tuple(sorted(pool[int(i)] for i in picked))


def generate_trials(catalog: Catalog, split: str, seed: int, k: int, subjects: int | None = None) -> TrialPlan:
    """The ordered enrollment and trial lists of one split (Req 13.1-13.5, 13.9).

    Trial IDs follow probe key order; each probe's genuine trial comes first, then its impostors
    in subject order. Never uses Real samples as probes (Req 13.4).
    """
    if k < 0:
        raise ValueError(f"impostors per probe must be >= 0, got {k}")
    labeled = list(catalog.iter_labeled(split))
    chosen = set(select_subjects((lab.subject_id for _, lab in labeled), subjects, seed))

    enrollments: list[Enrollment] = []
    enrolled: set[Identity] = set()
    for key, lab in labeled:
        if lab.is_real and lab.subject_id in chosen and lab.identity not in enrolled:
            enrolled.add(lab.identity)
            enrollments.append(Enrollment(f"e{len(enrollments) + 1:05d}", key, lab.identity))

    # Candidate impostor claims per finger position, sorted by subject.
    by_position: dict[tuple[str, str], list[Identity]] = defaultdict(list)
    for ident in sorted(enrolled, key=lambda i: (i.hand.value, i.finger.value, i.user_id)):
        by_position[(ident.hand.value, ident.finger.value)].append(ident)

    trials: list[Trial] = []
    skipped = 0

    def add(key: str, lab: Any, claim: Identity, genuine: bool) -> None:
        trials.append(Trial(f"t{len(trials) + 1:07d}", key, lab.identity, claim, genuine,
                            lab.alteration, lab.difficulty))

    for key, lab in labeled:
        if lab.is_real or lab.subject_id not in chosen:
            continue
        if lab.identity not in enrolled:
            skipped += 1
            continue
        add(key, lab, lab.identity, True)
        candidates = [c for c in by_position[(lab.identity.hand.value, lab.identity.finger.value)]
                      if c.user_id != lab.subject_id]
        if len(candidates) > k:
            rng = np.random.default_rng([seed, _stable_int(key)])
            idx = sorted(int(i) for i in rng.choice(len(candidates), size=k, replace=False))
            candidates = [candidates[i] for i in idx]
        for claim in candidates:
            add(key, lab, claim, False)

    return TrialPlan(split, seed, k, tuple(sorted(chosen)), tuple(enrollments), tuple(trials), skipped)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class EnrollResult:
    enroll_id: str
    run_id: str
    sample: str
    user_id: int
    hand: str
    finger: str
    status: str
    reason: str | None
    quality_before: int | None
    attempts_used: int
    minutiae_counts: str  # adapted count per attempt, ";"-separated ("" = not run)
    quality_after: str    # quality_after per attempt, ";"-separated
    template_id: str | None


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial_id: str
    run_id: str
    probe_sample: str
    probe_subject: int
    claim_user: int
    claim_hand: str
    claim_finger: str
    genuine: bool
    alteration: str
    difficulty: str
    status: str
    reason: str | None
    score: int | None
    threshold: int
    quality_before: int | None
    attempts_used: int
    minutiae_counts: str
    quality_after: str
    cache_hit: bool


def _joined(values: Iterable[int | None]) -> str:
    return ";".join("" if v is None else str(v) for v in values)


# --------------------------------------------------------------------------- execution


class MemoryProbeCache:
    """In-process `ProbeCache` (Req 13.6); `hits` counts successful lookups."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], CachedProbe] = {}
        self.hits = 0

    def get(self, key: tuple[str, str]) -> CachedProbe | None:
        value = self._data.get(key)
        if value is not None:
            self.hits += 1
        return value

    def put(self, key: tuple[str, str], value: CachedProbe) -> None:
        self._data[key] = value


StagesFactory = Callable[["PipelineConfig"], "Stages"]


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Everything a worker needs; pickled once per worker process.

    The config travels as its path (PipelineConfig holds read-only mappings, which do not
    pickle); workers reload it and check the `config_set_id`.
    """

    config_path: str
    config_root: str
    config_set_id: str
    catalog: Catalog
    store_root: str
    split: str
    pipeline_version: str
    stages_factory: StagesFactory | None = None  # None: LocalStages (tests inject fakes)


@dataclass(frozen=True, slots=True)
class WorkUnit:
    group: str  # "<Hand>_<finger>"
    enrollments: tuple[Enrollment, ...]
    trials: tuple[Trial, ...]


def build_units(plan: TrialPlan) -> list[WorkUnit]:
    """One unit per finger position, largest first (deterministic; balances the pool)."""
    enr: dict[str, list[Enrollment]] = defaultdict(list)
    tri: dict[str, list[Trial]] = defaultdict(list)
    for e in plan.enrollments:
        enr[f"{e.identity.hand.value}_{e.identity.finger.value}"].append(e)
    for t in plan.trials:
        tri[f"{t.claim.hand.value}_{t.claim.finger.value}"].append(t)
    groups = sorted(set(enr) | set(tri))
    units = [WorkUnit(g, tuple(enr[g]), tuple(tri[g])) for g in groups]
    return sorted(units, key=lambda u: (-(len(u.enrollments) + len(u.trials)), u.group))


_W: dict[str, Any] = {}

# Same thread budget in every process, so results do not depend on the worker count.
_THREAD_ENV = {"OMP_NUM_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1"}


def _init_worker(ctx: EvalContext, conf: PipelineConfig | None = None) -> None:
    for name, value in _THREAD_ENV.items():
        os.environ.setdefault(name, value)
    from fpv import config as cfgmod
    from fpv.pipeline import LocalStages

    if conf is None:
        conf = cfgmod.load(ctx.config_path, root=Path(ctx.config_root))
    if conf.config_set_id != ctx.config_set_id:
        raise RuntimeError(f"worker loaded config_set_id {conf.config_set_id}, expected {ctx.config_set_id}")
    stages = ctx.stages_factory(conf) if ctx.stages_factory is not None else LocalStages(conf)
    _W.update(ctx=ctx, conf=conf, stages=stages)


def _run_unit(unit: WorkUnit) -> tuple[list[EnrollResult], list[TrialResult]]:
    ctx: EvalContext = _W["ctx"]
    conf: PipelineConfig = _W["conf"]
    base = Path(ctx.store_root) / unit.group
    raw_root = ctx.catalog.raw_root
    tag = f"eval-{ctx.split}-{conf.config_set_id}"
    cache = MemoryProbeCache()
    # The live DB sits on process-local temp storage and is moved into the store only once the unit
    # is done and checkpointed: a WAL DB on a bind mount is corrupted by any reader outside the
    # container (the host cannot share the WAL index), so the eval dir only ever holds complete DBs.
    work = Path(tempfile.mkdtemp(prefix=f"fpv-eval-{unit.group}-"))
    try:
        enroll_rows, trial_rows = _run_unit_in(unit, ctx, conf, cache, work / "fpv.sqlite", base, raw_root, tag)
        base.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "fpv.sqlite"), base / "fpv.sqlite")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return enroll_rows, trial_rows


def _run_unit_in(unit: WorkUnit, ctx: EvalContext, conf: PipelineConfig, cache: MemoryProbeCache, db: Path,
                 base: Path, raw_root: Path, tag: str) -> tuple[list[EnrollResult], list[TrialResult]]:
    from fpv.pipeline import PipelineController
    from fpv.storage import Repository

    enroll_rows: list[EnrollResult] = []
    trial_rows: list[TrialResult] = []
    with Repository(db, base / "templates", base / "probes", artifact_root=base) as repo:
        repo.migrate()
        ctl = PipelineController(conf, repo, ctx.catalog, _W["stages"], probe_cache=cache,
                                 pipeline_version=ctx.pipeline_version)
        for e in unit.enrollments:
            o = ctl.enroll(raw_root / e.sample, e.identity, session_id=f"{tag}-{e.enroll_id}",
                           presentation_no=1)
            enroll_rows.append(EnrollResult(
                e.enroll_id, o.run_id, e.sample, e.identity.user_id, e.identity.hand.value,
                e.identity.finger.value, o.status.value, o.reason, o.quality_before, len(o.attempts),
                _joined(a.minutiae_count for a in o.attempts), _joined(a.quality_after for a in o.attempts),
                o.template_id))
        for t in unit.trials:
            hits = cache.hits
            o = ctl.verify(raw_root / t.probe, t.claim, session_id=f"{tag}-{t.trial_id}", presentation_no=1)
            trial_rows.append(TrialResult(
                t.trial_id, o.run_id, t.probe, t.probe_identity.user_id, t.claim.user_id,
                t.claim.hand.value, t.claim.finger.value, t.genuine, t.alteration, t.difficulty,
                o.status.value, o.reason, o.score, o.threshold, o.quality_before, len(o.attempts),
                _joined(a.minutiae_count for a in o.attempts), _joined(a.quality_after for a in o.attempts),
                cache.hits > hits))
        # Fold the WAL into the main file and leave WAL mode, so the moved DB is self-contained
        # and safe to open from the host.
        repo.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        repo.connection.execute("PRAGMA journal_mode = DELETE")
    return enroll_rows, trial_rows


def run_plan(plan: TrialPlan, ctx: EvalContext, *, workers: int = 1, conf: PipelineConfig | None = None,
             progress: Callable[[str], None] | None = None) -> tuple[list[EnrollResult], list[TrialResult]]:
    """Run every unit of `plan` (in-process when `workers == 1`); results sorted by ID (Req 13.5).

    `conf` is reused by the in-process path; worker processes reload it from `ctx.config_path`.
    """
    if workers < 1:
        raise ValueError(f"--workers must be at least 1, got {workers}")
    store = Path(ctx.store_root)
    if store.exists():  # a fresh store per evaluation; only this harness writes here
        shutil.rmtree(store)
    store.mkdir(parents=True)
    units = build_units(plan)
    enrolls: list[EnrollResult] = []
    trials: list[TrialResult] = []
    t0 = time.monotonic()

    def collect(i: int, unit: WorkUnit, res: tuple[list[EnrollResult], list[TrialResult]]) -> None:
        enrolls.extend(res[0])
        trials.extend(res[1])
        if progress is not None:
            progress(f"  [{i}/{len(units)}] {unit.group}: {len(res[0])} enrollments, {len(res[1])} trials "
                     f"({time.monotonic() - t0:.0f}s)")

    if workers == 1 or len(units) <= 1:
        _init_worker(ctx, conf)
        for i, unit in enumerate(units, start=1):
            collect(i, unit, _run_unit(unit))
    else:
        with ProcessPoolExecutor(min(workers, len(units)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(ctx,)) as pool:
            futures = {pool.submit(_run_unit, u): u for u in units}
            for i, fut in enumerate(as_completed(futures), start=1):
                collect(i, futures[fut], fut.result())
    enrolls.sort(key=lambda r: r.enroll_id)
    trials.sort(key=lambda r: r.trial_id)
    return enrolls, trials


# --------------------------------------------------------------------------- metrics


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def fte(statuses: Iterable[str]) -> float:
    """Share of enrollment samples that did not reach ENROLLED (Req 13.8)."""
    s = list(statuses)
    return _ratio(sum(1 for x in s if x != Status.ENROLLED.value), len(s))


def fta(probe_rejected: Iterable[bool]) -> float:
    """Share of probes rejected for quality or minutiae (Req 13.8)."""
    r = list(probe_rejected)
    return _ratio(sum(r), len(r))


def fmr(impostor_scores: Sequence[int], threshold: int) -> float:
    return _ratio(sum(1 for s in impostor_scores if s >= threshold), len(impostor_scores))


def fnmr(genuine_scores: Sequence[int], threshold: int) -> float:
    return _ratio(sum(1 for s in genuine_scores if s < threshold), len(genuine_scores))


def generalized_fmr(fmr_value: float, fta_value: float) -> float:
    """`(1 - FTA) * FMR`; 0 when every probe failed to acquire (no comparison can false-match)."""
    return 0.0 if fta_value == 1.0 else (1.0 - fta_value) * fmr_value


def generalized_fnmr(fnmr_value: float, fta_value: float) -> float:
    """`FTA + (1 - FTA) * FNMR`; 1 when every probe failed to acquire."""
    return 1.0 if fta_value == 1.0 else fta_value + (1.0 - fta_value) * fnmr_value


@dataclass(frozen=True, slots=True)
class Metrics:
    probes: int
    rejected_probes: int
    fta: float
    genuine: int     # completed genuine comparisons
    impostor: int    # completed impostor comparisons
    fmr: float
    fnmr: float
    gfmr: float
    gfnmr: float
    not_enrolled: int
    errors: int


def compute_metrics(trials: Sequence[TrialResult], threshold: int) -> Metrics:
    """FTA over unique probes, FMR/FNMR over completed comparisons, and generalized rates."""
    rejected: dict[str, bool] = {}
    for t in trials:
        rejected[t.probe_sample] = rejected.get(t.probe_sample, False) or t.status in REJECTED
    gen = [t.score for t in trials if t.genuine and t.status in COMPLETED and t.score is not None]
    imp = [t.score for t in trials if not t.genuine and t.status in COMPLETED and t.score is not None]
    a = fta(rejected.values())
    m, n = fmr(imp, threshold), fnmr(gen, threshold)
    return Metrics(
        probes=len(rejected), rejected_probes=sum(rejected.values()), fta=a, genuine=len(gen),
        impostor=len(imp), fmr=m, fnmr=n, gfmr=generalized_fmr(m, a), gfnmr=generalized_fnmr(n, a),
        not_enrolled=sum(1 for t in trials if t.status == Status.NOT_ENROLLED.value),
        errors=sum(1 for t in trials if t.status == Status.ERROR.value))


def breakdowns(trials: Sequence[TrialResult], threshold: int) -> dict[str, list[tuple[str, Metrics]]]:
    """Metrics by alteration type, by difficulty, and by difficulty x alteration (Req 13.7)."""
    def sub(pred: Callable[[TrialResult], bool]) -> Metrics:
        return compute_metrics([t for t in trials if pred(t)], threshold)

    return {
        "alteration": [(a, sub(lambda t, a=a: t.alteration == a)) for a in ALTERATIONS],
        "difficulty": [(d, sub(lambda t, d=d: t.difficulty == d)) for d in DIFFICULTIES],
        "difficulty x alteration": [(f"{d} / {a}", sub(lambda t, d=d, a=a: t.difficulty == d and t.alteration == a))
                                    for d in DIFFICULTIES for a in ALTERATIONS],
    }


# --------------------------------------------------------------------------- DET plot


def det_points(genuine: Sequence[int], impostor: Sequence[int]) -> list[tuple[int, float, float]]:
    """`(threshold, fmr, fnmr)` for every integer threshold from 0 to max score + 1."""
    top = max([*genuine, *impostor, 0]) + 1
    return [(t, fmr(impostor, t), fnmr(genuine, t)) for t in range(top + 1)]


_PROBIT_TICKS = (0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99)


def plot_det(curves: Mapping[str, tuple[Sequence[int], Sequence[int]]], threshold: int, path: Path,
             title: str) -> None:
    """DET curves (FNMR vs FMR on normal-deviate axes) with the operating point at `threshold`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nd = NormalDist()
    lo, hi = _PROBIT_TICKS[0], _PROBIT_TICKS[-1]

    def z(p: float) -> float:
        return nd.inv_cdf(min(max(p, lo), hi))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, (gen, imp) in curves.items():
        if not gen or not imp:
            continue
        pts = det_points(gen, imp)
        ax.plot([z(p[1]) for p in pts], [z(p[2]) for p in pts], label=label, linewidth=1.4)
        ax.plot([z(fmr(imp, threshold))], [z(fnmr(gen, threshold))], "o", color=ax.lines[-1].get_color(),
                markersize=5)
    ticks = [z(p) for p in _PROBIT_TICKS]
    labels = [f"{p * 100:g}%" for p in _PROBIT_TICKS]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_xlabel("False match rate (FMR)")
    ax.set_ylabel("False non-match rate (FNMR)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, title=f"dots: threshold {threshold}", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- reports


def output_dir(conf: PipelineConfig, split: str) -> Path:
    return conf.path("reports") / "eval" / split / conf.config_set_id


def write_csv(path: Path, rows: Sequence[Any], cls: type) -> None:
    names = [f.name for f in fields(cls)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in asdict(r).items()})


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:.2f}%"


def _metrics_table(rows: Sequence[tuple[str, Metrics]], first: str) -> list[str]:
    out = [f"| {first} | probes | FTA | genuine | FNMR | impostor | FMR | gFNMR | gFMR | not enrolled | errors |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        out.append(f"| {name} | {m.probes} | {_pct(m.fta)} | {m.genuine} | {_pct(m.fnmr)} | {m.impostor} | "
                   f"{_pct(m.fmr)} | {_pct(m.gfnmr)} | {_pct(m.gfmr)} | {m.not_enrolled} | {m.errors} |")
    return out


def render_summary(plan: TrialPlan, conf: PipelineConfig, enrolls: Sequence[EnrollResult],
                   trials: Sequence[TrialResult], *, pipeline_version: str, workers: int,
                   elapsed_s: float | None) -> str:
    threshold = conf.thresholds.bozorth3_match
    overall = compute_metrics(trials, threshold)
    enrolled = sum(1 for e in enrolls if e.status == Status.ENROLLED.value)
    fte_value = fte(e.status for e in enrolls)
    gen = [t.score for t in trials if t.genuine and t.status in COMPLETED and t.score is not None]
    imp = [t.score for t in trials if not t.genuine and t.status in COMPLETED and t.score is not None]
    statuses = Counter((t.genuine, t.status) for t in trials)
    e_status = Counter((e.status, e.reason) for e in enrolls)
    cache_hits = sum(t.cache_hit for t in trials)

    lines = [
        f"# Evaluation: {plan.split} split",
        "",
        f"- config: `{conf.source_path}` (config_set_id `{conf.config_set_id}`, frozen: {conf.frozen})",
        f"- pipeline_version: `{pipeline_version}`",
        f"- thresholds: bozorth3_match {threshold}, nfiq2_min {conf.thresholds.nfiq2_min}, "
        f"minutiae_min {conf.thresholds.minutiae_min}",
        f"- seed {plan.seed}, impostors per probe {plan.k}, subjects {len(plan.subjects)}, workers {workers}"
        + (f", elapsed {elapsed_s:.0f} s" if elapsed_s is not None else ""),
        f"- enrollments {len(enrolls)}, trials {len(trials)} "
        f"({sum(t.genuine for t in trials)} genuine, {sum(not t.genuine for t in trials)} impostor), "
        f"probes skipped (finger not enrolled in the split) {plan.skipped_probes}, probe cache hits {cache_hits}",
        "- runs and artifacts: `store/<Hand>_<finger>/fpv.sqlite` (one store per finger position)",
        "",
        "## Overall at the active threshold",
        "",
        "| metric | value | counts |",
        "|---|---:|---|",
        f"| FTE | {_pct(fte_value)} | {len(enrolls) - enrolled} of {len(enrolls)} enrollments not ENROLLED |",
        f"| FTA | {_pct(overall.fta)} | {overall.rejected_probes} of {overall.probes} probes rejected |",
        f"| FMR | {_pct(overall.fmr)} | {sum(1 for s in imp if s >= threshold)} of {overall.impostor} "
        "impostor comparisons |",
        f"| FNMR | {_pct(overall.fnmr)} | {sum(1 for s in gen if s < threshold)} of {overall.genuine} "
        "genuine comparisons |",
        f"| generalized FMR | {_pct(overall.gfmr)} | (1 - FTA) x FMR |",
        f"| generalized FNMR | {_pct(overall.gfnmr)} | FTA + (1 - FTA) x FNMR |",
        "",
        f"Excluded from the rates: {overall.not_enrolled} NOT_ENROLLED trials (claimed finger failed to "
        f"enroll) and {overall.errors} ERROR trials.",
        "",
    ]
    if gen and imp:
        lines += [f"Score medians: genuine {float(np.median(gen)):g}, impostor {float(np.median(imp)):g}; "
                  f"impostor max {max(imp)}.", ""]
    for name, rows in breakdowns(trials, threshold).items():
        lines += [f"## By {name}", "", *_metrics_table(rows, name), ""]
    lines += ["## Status counts", "", "| trial | status | count |", "|---|---|---:|"]
    for (genuine, status), n in sorted(statuses.items(), key=lambda x: (not x[0][0], x[0][1])):
        lines.append(f"| {'genuine' if genuine else 'impostor'} | {status} | {n} |")
    lines += ["", "| enrollment status | reason | count |", "|---|---|---:|"]
    for (status, reason), n in sorted(e_status.items(), key=lambda x: (x[0][0], x[0][1] or "")):
        lines.append(f"| {status} | {reason or ''} | {n} |")
    lines += ["", "## DET", "", "![DET](det.png)", ""]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    out_dir: Path
    plan: TrialPlan
    enrollments: list[EnrollResult]
    trials: list[TrialResult]
    fte: float
    overall: Metrics


def evaluate(conf: PipelineConfig, catalog: Catalog, split: str, *, seed: int | None = None,
             k: int | None = None, subjects: int | None = None, workers: int = 1,
             out_dir: Path | None = None, stages_factory: StagesFactory | None = None,
             pipeline_version: str | None = None,
             progress: Callable[[str], None] | None = None) -> EvaluationReport:
    """Generate, run and report one split's evaluation (Req 13)."""
    from fpv.config import pipeline_version as _pipeline_version

    ev = conf.raw.get("evaluation", {}) or {}
    seed = int(ev.get("seed", 42)) if seed is None else seed
    k = int(ev.get("impostors_per_probe", 5)) if k is None else k
    out_dir = out_dir or output_dir(conf, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = pipeline_version or _pipeline_version()

    plan = generate_trials(catalog, split, seed, k, subjects)
    if progress is not None:
        progress(f"evaluate {split}: {len(plan.subjects)} subjects, {len(plan.enrollments)} enrollments, "
                 f"{len(plan.trials)} trials, {workers} worker(s) -> {out_dir}")
    ctx = EvalContext(str(Path(conf.source_path).resolve()), str(conf.root), conf.config_set_id, catalog,
                      str(out_dir / "store"), split, version, stages_factory)
    t0 = time.monotonic()
    enrolls, trials = run_plan(plan, ctx, workers=workers, conf=conf, progress=progress)
    elapsed = time.monotonic() - t0

    threshold = conf.thresholds.bozorth3_match
    write_csv(out_dir / "scores.csv", trials, TrialResult)
    write_csv(out_dir / "enrollments.csv", enrolls, EnrollResult)
    curves: dict[str, tuple[list[int], list[int]]] = {}

    def scores(sel: Callable[[TrialResult], bool]) -> tuple[list[int], list[int]]:
        done = [t for t in trials if sel(t) and t.status in COMPLETED and t.score is not None]
        return [t.score for t in done if t.genuine], [t.score for t in done if not t.genuine]

    curves["all"] = scores(lambda t: True)
    for d in DIFFICULTIES:
        curves[d] = scores(lambda t, d=d: t.difficulty == d)
    plot_det(curves, threshold, out_dir / "det.png", f"DET, {split} split ({conf.config_set_id})")
    (out_dir / "summary.md").write_text(
        render_summary(plan, conf, enrolls, trials, pipeline_version=version, workers=workers,
                       elapsed_s=elapsed), encoding="utf-8")
    return EvaluationReport(out_dir, plan, enrolls, trials, fte(e.status for e in enrolls),
                            compute_metrics(trials, threshold))

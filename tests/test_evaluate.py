"""Unit tests for the batch evaluation harness (Task 11, Req 13).

Trial generation on a synthetic catalog (3 subjects x 2 fingers), hand-computed metrics, and
execution through the real `PipelineController` with fake stages in 1 and 3 worker processes.
No toolchain or dataset needed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fpv import cli
from fpv import config as C
from fpv import evaluate as E
from fpv.adapter import XytMinutia, XytTemplate, read_xyt
from fpv.catalog import Catalog, parse_label
from fpv.matcher import MatchResult
from fpv.quality import QualityResult
from fpv.storage import Repository

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"
SHAPE = (103, 96)
FINGERS = (("M", "Left", "index"), ("M", "Right", "thumb"))
SUBJECTS = (1, 2, 3)
EXCLUDED_REAL = "Real/3__M_Right_thumb_finger.BMP"   # its probes have no enrollment
BAD_PROBE = "Altered/Altered-Hard/2__M_Left_index_finger_Zcut.BMP"  # fake NFIQ 2 rejects it


def synthetic_keys(subjects=SUBJECTS) -> list[str]:
    keys = []
    for s in subjects:
        for g, hand, finger in FINGERS:
            stem = f"{s}__{g}_{hand}_{finger}_finger"
            keys.append(f"Real/{stem}.BMP")
            for d in E.DIFFICULTIES:
                for a in E.ALTERATIONS:
                    keys.append(f"Altered/Altered-{d}/{stem}_{a}.BMP")
    return keys


def make_catalog(raw: Path) -> Catalog:
    split_of = {k: "train" for k in synthetic_keys()}
    split_of.update({k: "validation" for k in synthetic_keys((4,))})  # other split: never used
    split_files: dict[str, list[str]] = {}
    for k, s in split_of.items():
        split_files.setdefault(s, []).append(k)
    return Catalog({k.lower(): s for k, s in split_of.items()}, {EXCLUDED_REAL.lower(): True}, raw,
                   {s: tuple(sorted(v)) for s, v in split_files.items()})


@pytest.fixture
def catalog(tmp_path) -> Catalog:
    return make_catalog(tmp_path / "raw")


# --------------------------------------------------------------------------- trial generation


@pytest.mark.parametrize("k", [1, 5])
def test_trials_cover_the_protocol(catalog, k):
    plan = E.generate_trials(catalog, "train", seed=42, k=k)
    assert plan.subjects == SUBJECTS
    # Every non-excluded Real sample is enrolled; nothing from the validation split.
    assert [e.sample for e in plan.enrollments] == sorted(
        x for x in synthetic_keys() if x.startswith("Real/") and x != EXCLUDED_REAL)
    enrolled = {e.identity for e in plan.enrollments}
    assert len(enrolled) == 5

    assert all(not t.probe.startswith("Real/") for t in plan.trials)  # Req 13.4
    genuine = [t for t in plan.trials if t.genuine]
    impostor = [t for t in plan.trials if not t.genuine]
    # One genuine trial per Altered probe whose finger was enrolled (9 per enrolled finger).
    assert sorted(t.probe for t in genuine) == sorted(
        x for x in synthetic_keys() if x.startswith("Altered/") and parse_label(x).identity in enrolled)
    assert len(genuine) == 45 and plan.skipped_probes == 9
    assert {(t.probe, t.claim) for t in genuine}.isdisjoint({(t.probe, t.claim) for t in impostor})

    for t in plan.trials:
        lab = parse_label(t.probe)
        assert (t.probe_identity, t.alteration, t.difficulty) == (lab.identity, lab.alteration, lab.difficulty)
        assert t.genuine == (t.claim == t.probe_identity)
        assert t.claim in enrolled
    for t in impostor:
        assert (t.claim.hand, t.claim.finger) == (t.probe_identity.hand, t.probe_identity.finger)
        assert t.claim.user_id != t.probe_identity.user_id

    per_probe: dict[str, list] = {}
    for t in impostor:
        per_probe.setdefault(t.probe, []).append(t.claim)
    for probe in (t.probe for t in genuine):
        ident = parse_label(probe).identity
        available = {e for e in enrolled if (e.hand, e.finger) == (ident.hand, ident.finger)
                     and e.user_id != ident.user_id}
        claims = per_probe.get(probe, [])
        assert len(claims) == len(set(claims)) == min(k, len(available))
        assert set(claims) <= available

    ids = [t.trial_id for t in plan.trials]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_trial_lists_depend_only_on_the_seed(catalog):
    a = E.generate_trials(catalog, "train", seed=7, k=1)
    b = E.generate_trials(make_catalog(catalog.raw_root), "train", seed=7, k=1)
    c = E.generate_trials(catalog, "train", seed=8, k=1)
    assert a == b
    assert a.trials != c.trials
    assert [t.probe for t in a.trials if t.genuine] == [t.probe for t in c.trials if t.genuine]


def test_subject_subset_is_seeded(catalog):
    a = E.generate_trials(catalog, "train", seed=1, k=5, subjects=2)
    assert len(a.subjects) == 2 and a == E.generate_trials(catalog, "train", seed=1, k=5, subjects=2)
    chosen = set(a.subjects)
    assert all(e.identity.user_id in chosen for e in a.enrollments)
    assert all(t.probe_identity.user_id in chosen and t.claim.user_id in chosen for t in a.trials)
    subsets = {E.select_subjects(range(1, 101), 10, seed) for seed in range(5)}
    assert len(subsets) > 1 and all(len(s) == 10 for s in subsets)


# --------------------------------------------------------------------------- metrics


def _row(trial_id, probe, genuine, status, score=None, alteration="CR", difficulty="Easy") -> E.TrialResult:
    return E.TrialResult(trial_id, "run", probe, 1, 1, "Left", "index", genuine, alteration, difficulty, status,
                         None, score, 40, 30, 1, "20", "50", False)


def test_metrics_match_hand_computed_counts():
    rows = [
        _row("t1", "P1", True, "MATCH", 50),
        _row("t2", "P1", False, "NO_MATCH", 10),
        _row("t3", "P1", False, "MATCH", 40),          # score == threshold is a match
        _row("t4", "P2", True, "NO_MATCH", 30, "Zcut", "Hard"),
        _row("t5", "P2", False, "NO_MATCH", 5, "Zcut", "Hard"),
        _row("t6", "P3", True, "REJECTED_QUALITY", alteration="Obl"),
        _row("t7", "P3", False, "REJECTED_QUALITY", alteration="Obl"),
        _row("t8", "P4", True, "NOT_ENROLLED"),
        _row("t9", "P4", False, "ERROR"),
    ]
    m = E.compute_metrics(rows, 40)
    # 4 probes, 1 rejected; impostors 10, 40, 5 (1 >= 40); genuine 50, 30 (1 < 40).
    assert (m.probes, m.rejected_probes, m.genuine, m.impostor, m.not_enrolled, m.errors) == (4, 1, 2, 3, 1, 1)
    assert m.fta == pytest.approx(0.25)
    assert m.fmr == pytest.approx(1 / 3) and m.fnmr == pytest.approx(0.5)
    assert m.gfmr == pytest.approx(0.75 / 3)               # (1 - FTA) * FMR
    assert m.gfnmr == pytest.approx(0.25 + 0.75 * 0.5)     # FTA + (1 - FTA) * FNMR
    assert E.fmr([10, 40, 5], 41) == 0 and E.fnmr([50, 30], 51) == 1

    by = dict(E.breakdowns(rows, 40)["alteration"])
    assert (by["Obl"].fta, by["Obl"].gfnmr, by["Obl"].gfmr) == (1.0, 1.0, 0.0)
    assert math.isnan(by["Obl"].fmr)
    assert (by["Zcut"].fnmr, by["Zcut"].fmr, by["Zcut"].fta) == (1.0, 0.0, 0.0)
    assert dict(E.breakdowns(rows, 40)["difficulty"])["Hard"].probes == 1

    assert E.fte(["ENROLLED", "ENROLLED", "REJECTED_QUALITY", "ENROLLED"]) == pytest.approx(0.25)
    assert math.isnan(E.fte([])) and math.isnan(E.fta([]))


def test_det_points_are_monotone():
    pts = E.det_points([50, 30, 60], [5, 10, 40])
    assert pts[0] == (0, 1.0, 0.0) and pts[-1][1:] == (0.0, 1.0)
    assert all(a[1] >= b[1] and a[2] <= b[2] for a, b in zip(pts, pts[1:]))


# --------------------------------------------------------------------------- execution


class EvalFakeStages:
    """Deterministic content-driven stand-ins for every stage (picklable for worker processes)."""

    def __init__(self, conf=None):
        pass

    def assess_quality(self, image):
        return QualityResult(score=int(image.mean() / 2.55), actionable_feedback={}, raw_output="",
                             tool_version="test", duration_ms=0.0)

    def preprocess(self, image, attempt, configuration_id):
        return SimpleNamespace(attempt=attempt, original_shape=SHAPE, scale_factor=attempt.scale_factor,
                               configuration_id=configuration_id, timings_ms={}, image=image)

    def quality_after(self, pre):
        return QualityResult(score=55, actionable_feedback={}, raw_output="", tool_version="test",
                             duration_ms=0.0)

    def extract(self, pre):
        return SimpleNamespace(minutiae=(), count=20, configuration_id=pre.configuration_id, image=pre.image)

    def adapt(self, ex, pre):
        seed = int(hashlib.sha256(pre.image.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return XytTemplate(tuple(XytMinutia(int(x), int(y), int(t), 60) for x, y, t in
                                 zip(rng.integers(0, 90, 20), rng.integers(0, 90, 20), rng.integers(0, 360, 20))))

    def match(self, probe, gallery):
        digest = hashlib.sha256(Path(probe).read_bytes() + Path(gallery).read_bytes()).digest()
        return MatchResult(digest[0] % 100, read_xyt(probe).count, read_xyt(gallery).count, "test", 0.0)


def write_images(raw: Path) -> None:
    for key in synthetic_keys():
        p = raw / key
        p.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        img = rng.integers(60, 200, SHAPE, dtype=np.uint8)
        if key == BAD_PROBE:
            img[:] = 3  # fake NFIQ 2 score 1 < nfiq2_min
        assert cv2.imwrite(str(p), img)


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(CONFIG, root=PROJECT_ROOT)


def _run(tmp_path, conf, workers):
    raw = tmp_path / "raw"
    if not raw.exists():
        write_images(raw)
    out = tmp_path / f"out-{workers}"
    return E.evaluate(conf, make_catalog(raw), "train", seed=42, k=2, workers=workers, out_dir=out,
                      stages_factory=EvalFakeStages, pipeline_version="0.0.0+gtest")


def test_parallel_results_match_serial_and_cache_hits_are_logged(tmp_path, conf):
    serial = _run(tmp_path, conf, 1)
    parallel = _run(tmp_path, conf, 3)

    # Run and template IDs are random UUIDs; everything else must be identical.
    assert [replace(r, run_id="") for r in serial.trials] == [replace(r, run_id="") for r in parallel.trials]
    assert ([replace(r, run_id="", template_id="") for r in serial.enrollments]
            == [replace(r, run_id="", template_id="") for r in parallel.enrollments])
    assert serial.overall == parallel.overall
    assert [t.trial_id for t in parallel.trials] == sorted(t.trial_id for t in parallel.trials)
    # Left index: 27 probes x (1 genuine + 2 impostors); Right thumb: 18 probes x (1 + 1).
    assert len(parallel.trials) == len(parallel.plan.trials) == 27 * 3 + 18 * 2

    rows = {r.trial_id: r for r in parallel.trials}
    bad = [r for r in parallel.trials if r.probe_sample == BAD_PROBE]
    assert {r.status for r in bad} == {"REJECTED_QUALITY"} and parallel.overall.rejected_probes == 1
    assert all(r.status in ("MATCH", "NO_MATCH", "REJECTED_QUALITY") for r in parallel.trials)
    assert all(r.status == "ENROLLED" for r in parallel.enrollments)

    # The probe's first trial processes it; the next trial of the same probe replays the cache.
    first, second = [t for t in parallel.plan.trials if t.probe == parallel.plan.trials[0].probe][:2]
    assert (rows[first.trial_id].cache_hit, rows[second.trial_id].cache_hit) == (False, True)
    group = f"{first.claim.hand.value}_{first.claim.finger.value}"
    store = parallel.out_dir / "store" / group
    with Repository(store / "fpv.sqlite", store / "templates", store / "probes", artifact_root=store) as repo:
        ev1 = repo.stage_events(rows[first.trial_id].run_id)
        ev2 = repo.stage_events(rows[second.trial_id].run_id)
        assert repo.get_run(rows[second.trial_id].run_id)["status"] == rows[second.trial_id].status
    q1 = next(e for e in ev1 if e["stage"] == "quality_before")
    q2 = next(e for e in ev2 if e["stage"] == "quality_before")
    assert q1["details"]["cache_hit"] is False and q1["outcome"] == "ok"
    assert q2["details"]["cache_hit"] is True and q2["outcome"] == "skipped"
    assert "preprocess" not in [e["stage"] for e in ev2]

    for name in ("scores.csv", "enrollments.csv", "summary.md", "det.png"):
        assert (parallel.out_dir / name).stat().st_size > 0
    header = (parallel.out_dir / "scores.csv").read_text().splitlines()[0].split(",")
    assert header[:12] == ["trial_id", "run_id", "probe_sample", "probe_subject", "claim_user", "claim_hand",
                           "claim_finger", "genuine", "alteration", "difficulty", "status", "reason"]
    summary = (parallel.out_dir / "summary.md").read_text()
    for text in ("FTE", "FTA", "generalized FMR", "## By alteration", "## By difficulty", "Zcut", "Hard"):
        assert text in summary


# --------------------------------------------------------------------------- CLI


def test_cli_test_split_requires_a_frozen_config():
    out, err = [], []
    code = cli.main(["--config", str(CONFIG), "evaluate", "--split", "test"], out=out.append, err=err.append)
    assert code == cli.EXIT_USAGE and "frozen" in err[-1] and not out


def test_cli_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--split", "dev"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--split", "train", "--workers", "0"])
    args = cli.build_parser().parse_args(["evaluate", "--split", "train", "--subjects", "50", "--workers", "4",
                                          "--seed", "3"])
    assert (args.subjects, args.workers, args.seed, args.func) == (50, 4, 3, cli.cmd_evaluate)

"""Unit tests for threshold calibration, the config freeze and the test-split guard (Task 12, Req 14).

Synthetic score tables stand in for `fpv evaluate` outputs; the guard test runs the real
evaluation harness with the fake stages of `test_evaluate`. No toolchain or dataset needed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import test_evaluate as TE
from fpv import calibrate as K
from fpv import cli
from fpv import config as C
from fpv import evaluate as E
from fpv.catalog import Catalog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"


def provisional_raw(reports: Path) -> dict:
    raw = copy.deepcopy(dict(C.load(CONFIG, root=PROJECT_ROOT).raw))
    raw["paths"] = {**raw["paths"], "reports": str(reports)}
    return raw


def write_config(path: Path, raw: dict) -> C.PipelineConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return C.load(path, root=path.parent)


# --------------------------------------------------------------------------- operating points


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("target", [0.001, 0.01, 0.05])
def test_threshold_is_minimal_and_meets_the_target(seed, target):
    rng = np.random.default_rng(seed)
    impostor = rng.geometric(0.08, size=int(rng.integers(2_000, 20_000))).tolist()
    genuine = np.clip(rng.normal(150, 80, size=3_000), 0, None).astype(int).tolist()
    op = K.threshold_at(genuine, impostor, target)
    assert E.fmr(impostor, op.threshold) <= target
    assert op.threshold == 0 or E.fmr(impostor, op.threshold - 1) > target  # minimal
    assert op.fmr == E.fmr(impostor, op.threshold) and op.fnmr == E.fnmr(genuine, op.threshold)
    assert (op.impostor, op.genuine) == (len(impostor), len(genuine))


def test_threshold_hand_computed():
    impostor = list(range(100))           # FMR(t) = (100 - t) / 100
    genuine = [50, 98, 99, 100, 200]
    op = K.threshold_at(genuine, impostor, 0.01)
    assert (op.threshold, op.false_matches, op.fmr) == (99, 1, 0.01)
    assert (op.false_non_matches, op.fnmr) == (2, 0.4)
    assert K.threshold_at(genuine, impostor, 0.001).threshold == 100  # above every impostor
    assert K.threshold_at([5], [0, 0, 0], 0.5).threshold == 1
    with pytest.raises(ValueError):
        K.threshold_at(genuine, [], 0.01)


def test_low_confidence_flag_on_small_impostor_sets():
    assert (K.min_impostors(0.001), K.min_impostors(0.01)) == (10_000, 1_000)
    small, enough = list(range(999)), list(range(1_000))
    assert K.threshold_at([1], small, 0.01).low_confidence
    assert not K.threshold_at([1], enough, 0.01).low_confidence
    assert K.threshold_at([1], enough, 0.001).low_confidence
    assert not K.threshold_at([1], list(range(10_000)), 0.001).low_confidence


# --------------------------------------------------------------------------- synthetic evaluation outputs


def synthetic_split(rng, split: str, subjects: range, *, k: int = 5):
    """Enrollment and trial rows shaped like `fpv evaluate` output (one finger per subject)."""
    enrolls, trials = [], []
    for s in subjects:
        q = int(rng.integers(0, 60))
        counts = "11;13" if s % 7 == 0 else ("9;9;9" if s % 11 == 0 else str(int(rng.integers(12, 40))))
        status = ("REJECTED_QUALITY" if q < 5 else "REJECTED_MINUTIAE" if counts == "9;9;9" else "ENROLLED")
        enrolls.append(E.EnrollResult(f"e{s:05d}", f"r{s}", f"Real/{s}__M_Left_index_finger.BMP", s, "Left",
                                      "index", status, None if status == "ENROLLED" else "x", q,
                                      len(counts.split(";")), counts, "", None))
    for s in subjects:
        for d in E.DIFFICULTIES:
            probe = f"Altered/Altered-{d}/{s}__M_Left_index_finger_CR.BMP"
            q = int(rng.integers(0, 60))
            rejected = q < 5
            claims = [(s, True)] + [(int(c), False) for c in rng.choice([x for x in subjects if x != s], k,
                                                                        replace=False)]
            for claim, genuine in claims:
                if rejected:
                    status, score = "REJECTED_QUALITY", None
                else:
                    score = int(rng.normal(120, 60)) if genuine else int(rng.geometric(0.1))
                    score = max(score, 0)
                    status = "MATCH" if score >= 40 else "NO_MATCH"
                trials.append(E.TrialResult(
                    f"t{len(trials) + 1:07d}", "run", probe, s, claim, "Left", "index", genuine, "CR", d, status,
                    "below_threshold" if rejected else None, score, 40, q, 0 if rejected else 1,
                    "" if rejected else "25", "" if rejected else "50", claim != s))
    return enrolls, trials


def write_eval(reports: Path, split: str, cs: str, enrolls, trials) -> Path:
    d = reports / "eval" / split / cs
    d.mkdir(parents=True, exist_ok=True)
    E.write_csv(d / "scores.csv", trials, E.TrialResult)
    E.write_csv(d / "enrollments.csv", enrolls, E.EnrollResult)
    return d


@pytest.fixture
def setup(tmp_path):
    reports = tmp_path / "reports"
    conf = write_config(tmp_path / "configs" / "pipeline.v1.yaml", provisional_raw(reports))
    rng = np.random.default_rng(7)
    data = {"train": synthetic_split(rng, "train", range(1, 301)),
            "validation": synthetic_split(rng, "validation", range(301, 401))}
    for split, (enr, tri) in data.items():
        write_eval(reports, split, conf.config_set_id, enr, tri)
    # A test-split output that must never be read: unparseable as a score table.
    test_dir = reports / "eval" / "test" / conf.config_set_id
    test_dir.mkdir(parents=True)
    (test_dir / "scores.csv").write_text("not,a\nscore table\n")
    (test_dir / "enrollments.csv").write_text("garbage")
    return conf, data, test_dir


# --------------------------------------------------------------------------- inputs


def test_calibration_never_reads_the_test_split(setup):
    conf, data, test_dir = setup
    with pytest.raises(K.CalibrationInputError):
        K.check_calibration_input(test_dir / "scores.csv")
    with pytest.raises(K.CalibrationInputError):
        K.load_trials(test_dir / "scores.csv")
    with pytest.raises(K.CalibrationInputError):
        K.check_calibration_input(Path("/elsewhere/train/scores.csv"))  # not under eval/<split>/<cs>/
    eval_root = conf.path("reports") / "eval"
    all_csv = sorted(eval_root.glob("*/*/scores.csv"))
    assert len(all_csv) == 3
    assert [p.parent.parent.name for p in K.filter_calibration_inputs(all_csv)] == ["train", "validation"]
    assert list(K.calibration_inputs(conf)) == ["train", "validation"]

    res = K.calibrate(conf)  # would fail on the garbage test table if it were read
    assert [s.split for s in res.splits] == ["train", "validation"]
    gen, imp = K.completed_scores([t for _, tri in data.values() for t in tri])
    assert res.primary == K.threshold_at(gen, imp, 0.01)
    assert res.primary_target_fmr == 0.01 and [op.target_fmr for op in res.pooled] == [0.001, 0.01]


def test_calibration_requires_both_splits(tmp_path):
    reports = tmp_path / "reports"
    conf = write_config(tmp_path / "c.yaml", provisional_raw(reports))
    rng = np.random.default_rng(1)
    write_eval(reports, "train", conf.config_set_id, *synthetic_split(rng, "train", range(1, 20)))
    with pytest.raises(K.CalibrationInputError, match="validation"):
        K.calibrate(conf)


# --------------------------------------------------------------------------- sweeps


def test_sweep_at_the_evaluated_gates_reproduces_recorded_rates(setup):
    conf, data, _ = setup
    res = K.calibrate(conf)
    q0, m0 = conf.thresholds.nfiq2_min, conf.thresholds.minutiae_min
    fte_cell = K._cell(res.sweep, "FTE", q0, m0)
    fta_cell = K._cell(res.sweep, "FTA", q0, m0)
    assert fte_cell.exact and fta_cell.exact
    assert fte_cell.low == pytest.approx(res.fte_at_primary["pooled"])
    assert fta_cell.low == pytest.approx(res.metrics_at_primary["pooled"].fta)
    # Rates never decrease as either gate gets stricter.
    for kind in ("FTE", "FTA"):
        lows = [K._cell(res.sweep, kind, q, m0).low for q in K.NFIQ2_CUTOFFS]
        assert lows == sorted(lows)


def test_acquisition_outcome_replays_the_attempt_policy():
    def out(q, counts, nq=5, m=12):
        a = K.Acquisition("ENROLLED", None, q, counts)
        return K.acquisition_outcome(a, nq, m, evaluated_nfiq2_min=5, max_attempts=3)

    assert out(30, (25,)) == "ok"
    assert out(30, (25,), m=26) == "unknown"          # attempt 2 never ran
    assert out(30, (11, 13), m=11) == "ok"             # attempt 1 would have been accepted
    assert out(30, (11, 13), m=14) == "unknown"
    assert out(30, (9, 9, 9)) == "rejected"            # all attempts ran
    assert out(30, (9, 9, 9), m=10) == "rejected"
    assert out(30, (25,), nq=35) == "rejected"         # stricter raw gate
    assert out(3, (), nq=0) == "unknown"               # never processed at the evaluated gate
    assert out(None, ()) == "rejected"                 # NFIQ 2 could not score it
    err = K.Acquisition("ERROR", "nfiq2_tool_error", None, ())
    assert K.acquisition_outcome(err, 5, 12, evaluated_nfiq2_min=5, max_attempts=3) == "error"


# --------------------------------------------------------------------------- freeze


def test_frozen_config_loads_validates_and_gets_a_new_id(setup):
    conf, _, _ = setup
    res, frozen, report = K.run_calibration(conf, pipeline_version="0.1.0+gtest")
    path = conf.root / C.FROZEN_CONFIG
    assert frozen.source_path == path and path.is_file() and report.is_file()
    again = C.load(path, root=conf.root)
    C.validate(yaml.safe_load(path.read_text()))
    assert again.frozen is True and again.config_set_id == frozen.config_set_id != conf.config_set_id
    assert again.thresholds.bozorth3_match == res.primary.threshold
    assert again.thresholds.nfiq2_min == conf.thresholds.nfiq2_min
    cal = again.raw["calibration"]
    assert cal["primary_target_fmr"] == 0.01
    assert [o["bozorth3_match"] for o in cal["operating_points"]] == [op.threshold for op in res.pooled]
    prov = again.raw["provenance"]
    assert prov["source_config_set_id"] == conf.config_set_id and prov["pipeline_version"] == "0.1.0+gtest"
    assert [r["split"] for r in prov["source_runs"]] == ["train", "validation"]
    assert all(len(r["sha256"]["scores.csv"]) == 64 for r in prov["source_runs"])
    assert prov["calibrated_at"]
    # Provenance is not hashed: editing it keeps the id.
    edited = dict(again.raw, provenance={"calibrated_at": "other"})
    assert C.config_set_id(edited) == again.config_set_id
    js = json.loads((report.parent / "calibration.json").read_text())
    assert js["frozen_config_set_id"] == frozen.config_set_id and js["sweep"]

    with pytest.raises(K.CalibrationInputError, match="already exists"):
        K.run_calibration(conf)
    with pytest.raises(K.CalibrationInputError, match="already frozen"):
        K.calibrate(frozen)
    _, changed, _ = K.run_calibration(conf, nfiq2_min=10, force=True)
    assert changed.thresholds.nfiq2_min == 10 and changed.config_set_id != frozen.config_set_id


def test_cli_calibrate_writes_report_and_frozen_config(setup):
    conf, _, _ = setup
    out, err = [], []
    assert cli.main(["--config", str(conf.source_path), "calibrate", "--out", str(conf.root / "f.yaml")],
                    out=out.append, err=err.append) == 0
    assert any("target FMR 0.01 (primary)" in line for line in out)
    assert (K.calibration_dir(conf) / "calibration.md").is_file()
    assert C.load(conf.root / "f.yaml", root=conf.root).frozen
    code = cli.main(["--config", str(conf.source_path), "calibrate", "--out", str(conf.root / "f.yaml")],
                    out=out.append, err=err.append)
    assert code == cli.EXIT_USAGE and "already exists" in err[-1]


# --------------------------------------------------------------------------- CLI default config


def test_cli_picks_the_frozen_config_by_default(tmp_path, monkeypatch):
    raw = provisional_raw(tmp_path / "reports")
    prov = write_config(tmp_path / C.DEFAULT_CONFIG, raw)
    monkeypatch.chdir(tmp_path)

    def startup(argv):
        err = []
        cli.main([*argv, "match", "missing.xyt", "missing.xyt"], out=lambda s: None, err=err.append)
        return err[0]

    assert str(C.DEFAULT_CONFIG) in startup([]) and prov.config_set_id in startup([])
    frozen = write_config(tmp_path / C.FROZEN_CONFIG, dict(copy.deepcopy(raw), frozen=True))
    line = startup([])
    assert str(C.FROZEN_CONFIG) in line and frozen.config_set_id in line and "frozen" in line
    assert str(C.DEFAULT_CONFIG) in startup(["--config", str(C.DEFAULT_CONFIG)])


# --------------------------------------------------------------------------- test-split guard


def test_test_split_guard(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    TE.write_images(raw_dir)
    keys = TE.synthetic_keys()
    catalog = Catalog({k.lower(): "test" for k in keys}, {TE.EXCLUDED_REAL.lower(): True}, raw_dir,
                      {"test": tuple(sorted(keys))})
    monkeypatch.setattr(Catalog, "load", classmethod(lambda cls, *a, **kw: catalog))
    real_evaluate = E.evaluate
    monkeypatch.setattr(E, "evaluate", lambda conf, cat, split, **kw: real_evaluate(
        conf, cat, split, **{**kw, "workers": 1, "k": 2, "stages_factory": TE.EvalFakeStages}))

    reports = tmp_path / "reports"
    raw = provisional_raw(reports)
    prov = write_config(tmp_path / "prov.yaml", raw)
    frozen_raw = dict(copy.deepcopy(raw), frozen=True)
    frozen_raw["calibration"] = {"target_fmrs": [0.001, 0.01], "primary_target_fmr": 0.01,
                                 "operating_points": [{"target_fmr": 0.001, "bozorth3_match": 90},
                                                      {"target_fmr": 0.01, "bozorth3_match": 40}]}
    frozen = write_config(tmp_path / "frozen.yaml", frozen_raw)

    def run(conf, *extra):
        out, err = [], []
        code = cli.main(["--config", str(conf.source_path), "evaluate", "--split", "test", *extra],
                        out=out.append, err=err.append)
        return code, out, err

    code, out, err = run(prov)  # not frozen
    assert code == cli.EXIT_USAGE and "frozen" in err[-1] and not out

    code, out, _ = run(frozen)
    report_path = reports / f"test_report_{frozen.config_set_id}.json"
    assert code == 0 and report_path.is_file()
    first = json.loads(report_path.read_text())
    assert first["forced"] is False and first["override"] is None and first["config_set_id"] == frozen.config_set_id
    assert [o["bozorth3_match"] for o in first["operating_points"]] == [90, 40]
    baseline = (reports / "baseline_v1.md").read_text()
    for text in ("FTE", "FTA", "FMR", "FNMR", "## By alteration", "## By difficulty", "Zcut", "Hard",
                 frozen.config_set_id):
        assert text in baseline

    code, out, err = run(frozen)  # second run without --force
    assert code == cli.EXIT_USAGE and "--force" in err[-1] and not out
    assert json.loads(report_path.read_text()) == first

    code, out, _ = run(frozen, "--force")
    assert code == 0
    second = json.loads(report_path.read_text())
    assert second["forced"] is True and second["override"]["forced"] is True
    assert second["override"]["replaced_report_completed_at"] == first["completed_at"]
    assert len(second["history"]) == 1 and second["history"][0]["forced"] is False
    assert "--force" in (reports / "baseline_v1.md").read_text()

    code, _, err = run(frozen, "--force", "--subjects", "2")
    assert code == cli.EXIT_USAGE and "--subjects" in err[-1]

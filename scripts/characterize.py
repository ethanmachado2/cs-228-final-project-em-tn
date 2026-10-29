#!/usr/bin/env python3
"""Task 5 characterization study (Req 6.4): evidence for the attempt policy and thresholds.

For a seeded sample of train-split fingers it takes the Real sample and one seeded Altered
sample per difficulty (Easy, Medium, Hard), then for each sample:

* scores the raw image with NFIQ 2 at the configured 3x pre-scoring upscale and at 2x;
* runs every preprocessing *chain* (the configured attempts plus study variants), scoring
  `quality_after` (3x for every chain, 2x too for the configured attempts);
* runs SBMEX on each chain with `minimum_background_distance` in MBD_VALUES and records the
  raw and deduplicated minutiae counts and per-stage timings.

Chains use the two AttemptConfig resolution knobs:

* `upscale` u: bicubic upscale with pyfing told dpi = 500*u (pyfing resamples back to 500 dpi
               internally for every DNN/filter step). The only knob before this study.
* `resample` r: bicubic upscale with the result treated as a native 500 ppi image (pyfing told
               dpi = 500). Added by this study (Task 5) and used by the final attempts.

Output: data/quality_reports/characterization/{characterization.csv, summary.md}.
Usage (container): python scripts/characterize.py --n 200 --seed 42 [--workers 4]
Re-summarize an existing CSV: python scripts/characterize.py --summarize-only

The aggregation functions (`summarize`, `policy_table`, `render_summary`) are pure pandas so
they can be unit-tested on a tiny fake table without the toolchain.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LEVELS = ("Real", "Easy", "Medium", "Hard")
ALTERATIONS = ("Obl", "CR", "Zcut")
MBD_VALUES = (4, 6, 8, 10)
DEFAULT_MBD = 10                       # SbmexParameters.minimum_background_distance default
NFIQ2_CUTOFFS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
MINUTIAE_MINS = (10, 12, 15, 20)
STEP_COLS = ("t_segmentation", "t_orientation", "t_frequency", "t_enhancement")
OUT_DIR = Path("data/quality_reports/characterization")


@dataclass(frozen=True)
class Chain:
    """One preprocessing variant. `attempt_no` set => the configured attempt, used verbatim."""

    name: str
    attempt_no: int | None = None
    deep: bool = True
    clahe: bool = False
    upscale: float = 1.0               # `dpi` semantics (pyfing dpi = 500 * upscale)
    resample: float = 1.0              # `resample` semantics (pyfing dpi = 500)


# A1-A3 are whatever configs/pipeline.v1.yaml defines (after this study: 3x resample chains).
# The dpi_* chains reproduce the pre-study attempt definitions (Task 4 provisional config).
CHAINS: tuple[Chain, ...] = (
    Chain("A1", attempt_no=1),
    Chain("A2", attempt_no=2),
    Chain("A3", attempt_no=3),
    Chain("dpi_trad_1x", deep=False),                         # pre-study attempt 1
    Chain("dpi_deep_1x"),                                     # pre-study attempt 2
    Chain("dpi_deep_clahe_up2", clahe=True, upscale=2.0),     # pre-study attempt 3 (MBD 6)
    Chain("dpi_deep_up2", upscale=2.0),                       # Task guidance: 2x upscale, default MBD
    Chain("trad_rs2", deep=False, resample=2.0),
    Chain("deep_rs2", resample=2.0),
)


# ============================================================================ sampling


@dataclass(frozen=True)
class SampleTask:
    key: str                # dataset key, e.g. Real/100__M_Left_index_finger.BMP
    finger_key: str         # e.g. 100__M_Left_index_finger
    difficulty: str
    alteration: str


def select_samples(catalog: Any, raw_root: Path, n: int, seed: int) -> list[SampleTask]:
    """Seeded choice of n train-split Real fingers plus one Altered sample per difficulty."""
    reals = sorted(r.source_path for r in catalog.iter_samples("train", real=True))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(reals), size=min(n, len(reals)), replace=False).tolist())
    tasks: list[SampleTask] = []
    for idx in chosen:
        real_key = reals[idx]
        name = Path(real_key).name
        stem, ext = name.rsplit(".", 1)
        tasks.append(SampleTask(real_key, stem, "Real", "Real"))
        for level in LEVELS[1:]:
            cands = []
            for alt in ALTERATIONS:
                key = f"Altered/Altered-{level}/{stem}_{alt}.{ext}"
                if (raw_root / key).exists() and catalog.split_of(key) == "train" \
                        and not catalog.is_excluded(key):
                    cands.append((key, alt))
            if cands:  # rng is consumed only when there is a choice, so the draw stays reproducible
                key, alt = cands[int(rng.integers(len(cands)))]
                tasks.append(SampleTask(key, stem, level, alt))
    return tasks


# ============================================================================ worker

_W: dict[str, Any] = {}


def _init_worker(config_path: str, root: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from fpv import config as C
    from fpv.quality import QualitySettings

    conf = C.load(config_path, root=Path(root))
    _W["conf"] = conf
    _W["root"] = Path(root)
    _W["q3"] = QualitySettings.from_config(conf)                 # configured (3x)
    _W["q2"] = QualitySettings.from_config(conf, upscale=2.0)    # variant (2x)


def chain_attempt(chain: Chain, conf: Any) -> Any:
    from fpv.config import AttemptConfig, ClaheConfig

    if chain.attempt_no is not None:
        return conf.attempt(chain.attempt_no)
    if chain.deep:
        methods = dict(segmentation="SUFS", orientation="SNFOE", frequency="SNFFE", enhancement="SNFEN")
    else:
        methods = dict(segmentation="GMFS", orientation="GBFOE", frequency="XSFFE", enhancement="GBFEN")
    return AttemptConfig(attempt_no=9, name=chain.name, **methods,
                         clahe=ClaheConfig() if chain.clahe else None, upscale=chain.upscale,
                         resample=chain.resample)


def dedup_count(minutiae: Iterable[Any], scale: float, shape: tuple[int, int]) -> int:
    """Adapted (deduplicated) `.xyt` count, computed by the pipeline's own adapter."""
    from fpv.adapter import to_xyt

    return to_xyt(list(minutiae), shape, scale).count


def _score(img: np.ndarray, settings: Any) -> tuple[int | None, float]:
    from fpv.quality import assess_quality

    q = assess_quality(img, settings)
    return q.score, q.duration_ms


def process_sample(task: SampleTask) -> list[dict[str, Any]]:
    import dataclasses

    from fpv import imaging
    from fpv.extract import extract
    from fpv.preprocess import preprocess, quality_after_image

    conf, root = _W["conf"], _W["root"]
    raw = imaging.load_gray8(root / "data" / "raw" / task.key)
    shape = (int(raw.shape[0]), int(raw.shape[1]))
    raw_q3, t_raw_q3 = _score(raw, _W["q3"])
    raw_q2, _ = _score(raw, _W["q2"])
    base = dict(sample=task.key, finger_key=task.finger_key, difficulty=task.difficulty,
                alteration=task.alteration, raw_q3=raw_q3, raw_q2=raw_q2, t_raw_q3=t_raw_q3)
    rows: list[dict[str, Any]] = []
    for chain in CHAINS:
        attempt = chain_attempt(chain, conf)
        cfg_mbd = int(attempt.sbmex_overrides.get("minimum_background_distance", DEFAULT_MBD))
        scale = attempt.scale_factor
        common = dict(base, chain=chain.name, resample=attempt.resample, upscale=attempt.upscale,
                      scale=scale, configured_attempt=chain.attempt_no)
        try:
            t0 = time.perf_counter_ns()
            pre = preprocess(raw, attempt, configuration_id=conf.configuration_id(attempt))
            t_pre = (time.perf_counter_ns() - t0) / 1e6
            qa_img = quality_after_image(pre)  # back at the raw sample size
            qa3, t_qa3 = _score(qa_img, _W["q3"])
            qa2 = _score(qa_img, _W["q2"])[0] if chain.attempt_no is not None else None
        except Exception as exc:  # noqa: BLE001 - recorded, the study continues
            rows.append(dict(common, mbd=None, error=f"{type(exc).__name__}: {exc}"))
            continue
        timings = {f"t_{k}": v for k, v in pre.timings_ms.items()}
        for mbd in MBD_VALUES:
            overrides = {**attempt.sbmex_overrides, "minimum_background_distance": mbd}
            variant = dataclasses.replace(pre, attempt=dataclasses.replace(attempt, sbmex_overrides=overrides))
            try:
                ex = extract(variant)
                counts = dict(raw_count=ex.count, dedup_count=dedup_count(ex.minutiae, scale, shape),
                              t_extract=ex.duration_ms, error=None)
            except Exception as exc:  # noqa: BLE001
                counts = dict(raw_count=None, dedup_count=None, t_extract=None,
                              error=f"{type(exc).__name__}: {exc}")
            rows.append(dict(common, mbd=mbd, is_configured=chain.attempt_no is not None and mbd == cfg_mbd,
                             qa3=qa3, qa2=qa2, t_preprocess=t_pre, t_qa3=t_qa3, **timings, **counts))
    return rows


# ============================================================================ aggregation (pure)


def _pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _levels(df: pd.DataFrame) -> list[str]:
    return [lv for lv in LEVELS if lv in set(df["difficulty"])]


def samples_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample with its raw NFIQ 2 scores."""
    return df.drop_duplicates("sample")[["sample", "finger_key", "difficulty", "alteration", "raw_q3", "raw_q2"]]


def nfiq2_table(df: pd.DataFrame, cutoffs: Sequence[int] = NFIQ2_CUTOFFS) -> pd.DataFrame:
    """Raw NFIQ 2 distribution and gate pass rate per difficulty, for each pre-scoring upscale.

    An unscorable sample (NaN) fails every cutoff, as the gate does (Req 4.3).
    """
    s = samples_table(df)
    out = []
    for col, label in (("raw_q3", "3x"), ("raw_q2", "2x")):
        for lv in _levels(s) + ["All"]:
            v = s[col] if lv == "All" else s.loc[s["difficulty"] == lv, col]
            v = pd.to_numeric(v, errors="coerce")
            row = {"upscale": label, "difficulty": lv, "n": len(v), "unscorable": float(v.isna().mean())}
            q = v.dropna()
            for name, p in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75)):
                row[name] = float(q.quantile(p)) if len(q) else math.nan
            row["min"] = float(q.min()) if len(q) else math.nan
            for c in cutoffs:
                row[f"pass>={c}"] = float((v >= c).mean()) if len(v) else math.nan
            out.append(row)
    return pd.DataFrame(out)


def variant_table(df: pd.DataFrame, minutiae_mins: Sequence[int] = MINUTIAE_MINS) -> pd.DataFrame:
    """Per (chain, mbd, difficulty): median quality_after, dedup counts and per-attempt pass rates."""
    ok = df[df["mbd"].notna()]
    out = []
    for (chain, mbd), g in ok.groupby(["chain", "mbd"], sort=False):
        for lv in _levels(g) + ["All"]:
            h = g if lv == "All" else g[g["difficulty"] == lv]
            c = pd.to_numeric(h["dedup_count"], errors="coerce").fillna(0)
            qa = pd.to_numeric(h["qa3"], errors="coerce")
            row = {"chain": chain, "mbd": int(mbd), "difficulty": lv, "n": len(h),
                   "qa3_median": float(qa.median()) if qa.notna().any() else math.nan,
                   "qa3_unscorable": float(qa.isna().mean()),
                   "count_median": float(c.median()), "count_p10": float(c.quantile(.1))}
            for m in minutiae_mins:
                row[f"pass>={m}"] = float((c >= m).mean())
            out.append(row)
    return pd.DataFrame(out)


def quality_after_table(df: pd.DataFrame, cutoffs: Sequence[int] = (5, 10, 20)) -> pd.DataFrame:
    """quality_after of the configured attempts at the 3x and 2x NFIQ 2 pre-scoring upscale."""
    per = df[df["configured_attempt"].notna()].drop_duplicates(["sample", "chain"]) \
        if "configured_attempt" in df else df.iloc[0:0]
    out = []
    for chain, g in per.groupby("chain", sort=True):
        for col, label in (("qa3", "3x"), ("qa2", "2x")):
            for lv in _levels(g) + ["All"]:
                v = pd.to_numeric((g if lv == "All" else g[g["difficulty"] == lv])[col], errors="coerce")
                q = v.dropna()
                row = {"chain": chain, "upscale": label, "difficulty": lv, "n": len(v),
                       "unscorable": float(v.isna().mean()) if len(v) else math.nan,
                       "p10": float(q.quantile(.1)) if len(q) else math.nan,
                       "median": float(q.median()) if len(q) else math.nan}
                for c in cutoffs:
                    row[f">={c}"] = float((v >= c).mean()) if len(v) else math.nan
                out.append(row)
    return pd.DataFrame(out)


def _counts(df: pd.DataFrame, chain: str, mbd: int) -> pd.Series:
    g = df[(df["chain"] == chain) & (df["mbd"] == mbd)].set_index("sample")["dedup_count"]
    return pd.to_numeric(g, errors="coerce").fillna(0)


def policy_table(df: pd.DataFrame, policy: Sequence[tuple[str, int]], minutiae_min: int,
                 nfiq2_min: int | None = None) -> pd.DataFrame:
    """Per-attempt and cumulative acquisition rates for an ordered attempt policy.

    `policy` is [(chain, mbd), ...] in attempt order. A sample is acquired at the first attempt
    whose deduplicated count reaches `minutiae_min`. With `nfiq2_min`, samples failing the raw
    3x gate are never acquired (end-to-end rates); otherwise extraction-only rates.
    """
    s = samples_table(df).set_index("sample")
    gate = pd.Series(True, index=s.index)
    if nfiq2_min is not None:
        gate = pd.to_numeric(s["raw_q3"], errors="coerce").fillna(-1) >= nfiq2_min
    remaining = gate.copy()
    out = []
    for k, (chain, mbd) in enumerate(policy, start=1):
        c = _counts(df, chain, mbd).reindex(s.index).fillna(0)
        passes = c >= minutiae_min
        newly = remaining & passes
        remaining = remaining & ~passes
        for lv in [x for x in LEVELS if x in set(s["difficulty"])] + ["All"]:
            idx = s.index if lv == "All" else s.index[s["difficulty"] == lv]
            out.append({"attempt": k, "chain": chain, "mbd": mbd, "difficulty": lv, "n": len(idx),
                        "attempt_pass": float((gate[idx] & passes[idx]).mean()),
                        "cumulative": float((gate[idx] & ~remaining[idx]).mean()),
                        "newly_acquired": int(newly[idx].sum())})
    return pd.DataFrame(out)


def best_next_attempt(df: pd.DataFrame, prefix: Sequence[tuple[str, int]], minutiae_min: int,
                      candidates: Iterable[tuple[str, int]] | None = None) -> pd.DataFrame:
    """Rank candidate (chain, mbd) next attempts by the cumulative 'All' rate they add to `prefix`."""
    if candidates is None:
        candidates = sorted({(c, int(m)) for c, m in df.loc[df["mbd"].notna(), ["chain", "mbd"]]
                             .itertuples(index=False)})
    rows = []
    for cand in candidates:
        t = policy_table(df, [*prefix, cand], minutiae_min)
        last = t[(t["attempt"] == len(prefix) + 1)]
        row = {"chain": cand[0], "mbd": cand[1]}
        for _, r in last.iterrows():
            row[r["difficulty"]] = r["cumulative"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["All", "Hard"], ascending=False, kind="stable").reset_index(drop=True)


def timing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median per-stage timings (ms) per chain; extraction at the chain's first MBD row."""
    ok = df[df["mbd"].notna()]
    per_chain = ok.drop_duplicates(["sample", "chain"])
    cols = [c for c in (*STEP_COLS, "t_preprocess", "t_qa3") if c in per_chain]
    t = per_chain.groupby("chain", sort=False)[cols].median()
    t["t_extract"] = ok.groupby("chain", sort=False)["t_extract"].median()
    return t.reset_index()


@dataclass
class Summary:
    n_fingers: int
    n_samples: int
    errors: int
    nfiq2: pd.DataFrame
    quality_after: pd.DataFrame
    variants: pd.DataFrame
    policies: dict[str, pd.DataFrame]
    best_third: pd.DataFrame
    best_third_strict: pd.DataFrame
    strict_minutiae_min: int
    timings: pd.DataFrame


def summarize(df: pd.DataFrame, policies: Mapping[str, Sequence[tuple[str, int]]], *, minutiae_min: int,
              nfiq2_min: int, best_prefix: Sequence[tuple[str, int]],
              strict_minutiae_min: int = 20) -> Summary:
    pol = {}
    for name, p in policies.items():
        pol[name] = policy_table(df, p, minutiae_min)
        pol[name + " (end-to-end)"] = policy_table(df, p, minutiae_min, nfiq2_min=nfiq2_min)
    return Summary(
        n_fingers=int(df["finger_key"].nunique()),
        n_samples=int(df["sample"].nunique()),
        errors=int(df["error"].notna().sum()) if "error" in df else 0,
        nfiq2=nfiq2_table(df),
        quality_after=quality_after_table(df),
        variants=variant_table(df),
        policies=pol,
        best_third=best_next_attempt(df, best_prefix, minutiae_min),
        best_third_strict=best_next_attempt(df, best_prefix, strict_minutiae_min),
        strict_minutiae_min=strict_minutiae_min,
        timings=timing_table(df),
    )


def _md(t: pd.DataFrame, pct_cols: Iterable[str] = (), floatfmt: str = "{:.1f}") -> str:
    pct_cols = set(pct_cols)
    cols = list(t.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append(_pct(v))
            elif isinstance(v, float):
                cells.append("–" if math.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _policy_md(t: pd.DataFrame) -> str:
    wide_a = t.pivot_table(index=["attempt", "chain", "mbd"], columns="difficulty", values="attempt_pass",
                           sort=False).reset_index()
    wide_c = t.pivot_table(index=["attempt", "chain", "mbd"], columns="difficulty", values="cumulative",
                           sort=False).reset_index()
    lv = [c for c in [*LEVELS, "All"] if c in wide_a.columns]
    a = wide_a[["attempt", "chain", "mbd", *lv]]
    c = wide_c[["attempt", "chain", "mbd", *lv]]
    return ("Per attempt (pass rate of that attempt alone):\n\n" + _md(a, lv)
            + "\n\nCumulative (acquired by attempt k):\n\n" + _md(c, lv))


def render_summary(s: Summary, *, meta: Mapping[str, Any], findings: Sequence[str] = ()) -> str:
    out = [f"# Characterization study (Task 5)\n",
           "Generated by `scripts/characterize.py`. " + ", ".join(f"{k}={v}" for k, v in meta.items()) + ".",
           f"\n{s.n_fingers} train-split fingers, {s.n_samples} samples (Real + one seeded alteration per "
           f"Altered level), {s.errors} recorded errors. Pass rates count an unscorable sample as a failure.\n"]
    if findings:
        out += ["## Findings and decisions\n", *[f"- {f}" for f in findings], ""]
    nf = s.nfiq2
    pass_cols = [c for c in nf.columns if c.startswith("pass>=")]
    out += ["## Raw NFIQ 2 (quality_before): 3x vs 2x pre-scoring upscale\n",
            _md(nf, ["unscorable", *pass_cols], "{:.0f}"), ""]
    qa = s.quality_after
    if len(qa):
        qa_pct = ["unscorable", *[c for c in qa.columns if c.startswith(">=")]]
        out += ["## quality_after of the configured attempts: 3x vs 2x pre-scoring upscale\n",
                "Inverted, masked enhanced image resized back to the raw size, then scored like "
                "quality_before (diagnostic only, Req 5.6).\n", _md(qa, qa_pct, "{:.0f}"), ""]
    v = s.variants
    vp = [c for c in v.columns if c.startswith("pass>=")]
    out += ["## Chains × minimum_background_distance (deduplicated minutiae counts)\n",
            "`qa3` is quality_after at the 3x NFIQ 2 upscale. `pass>=m` is the share of samples whose "
            "deduplicated count reaches m on that variant alone.\n",
            _md(v, ["qa3_unscorable", *vp]), ""]
    for name, t in s.policies.items():
        out += [f"## Policy: {name}\n", _policy_md(t), ""]
    lv = [c for c in [*LEVELS, "All"] if c in s.best_third.columns]
    out += [f"## Best third attempt after the first two (cumulative by attempt 3, minutiae_min "
            f"{meta.get('minutiae_min', '')})\n", _md(s.best_third.head(15), lv), ""]
    lv2 = [c for c in [*LEVELS, "All"] if c in s.best_third_strict.columns]
    out += [f"## Best third attempt at a stricter floor ({s.strict_minutiae_min} minutiae)\n",
            "Separates the candidates when the first two attempts already acquire nearly everything.\n",
            _md(s.best_third_strict.head(15), lv2), ""]
    out += ["## Median stage timings (ms)\n", _md(s.timings), ""]
    return "\n".join(out) + "\n"


# ============================================================================ main

PRE_STUDY = [("dpi_trad_1x", 10), ("dpi_deep_1x", 10), ("dpi_deep_clahe_up2", 6)]


def policies_for(configured: list[tuple[str, int]]) -> dict[str, list[tuple[str, int]]]:
    a1, a2, a3 = configured
    return {
        "configured (A1, A2, A3)": configured,
        "configured, A3 at default MBD 10": [a1, a2, (a3[0], DEFAULT_MBD)],
        "pre-study config (dpi semantics: trad 1x, deep 1x, deep+CLAHE 2x upscale MBD 6)": PRE_STUDY,
        "pre-study, A3 -> deep 2x upscale, default MBD": [*PRE_STUDY[:2], ("dpi_deep_up2", 10)],
        "pre-study, A3 -> deep 1x, MBD 4": [*PRE_STUDY[:2], ("dpi_deep_1x", 4)],
        "pre-study, A3 -> deep 1x, MBD 6": [*PRE_STUDY[:2], ("dpi_deep_1x", 6)],
        "pre-study, A3 -> deep 1x, MBD 8": [*PRE_STUDY[:2], ("dpi_deep_1x", 8)],
        "resample 2x: trad 2x, deep 2x, then configured A3": [("trad_rs2", 10), ("deep_rs2", 10), a3],
    }


def configured_policy(conf: Any) -> list[tuple[str, int]]:
    return [(f"A{a.attempt_no}", int(a.sbmex_overrides.get("minimum_background_distance", DEFAULT_MBD)))
            for a in conf.attempts]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="number of train-split fingers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default="configs/pipeline.v1.yaml")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--minutiae-min", type=int, default=None, help="default: thresholds.minutiae_min")
    ap.add_argument("--nfiq2-min", type=int, default=None, help="default: thresholds.nfiq2_min")
    ap.add_argument("--summarize-only", action="store_true", help="re-render summary.md from the CSV")
    args = ap.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    from fpv import config as C

    conf = C.load(args.config, root=PROJECT_ROOT)
    minutiae_min = args.minutiae_min or conf.thresholds.minutiae_min
    nfiq2_min = conf.thresholds.nfiq2_min if args.nfiq2_min is None else args.nfiq2_min
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "characterization.csv"

    if not args.summarize_only:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        from fpv.catalog import Catalog

        catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
        tasks = select_samples(catalog, conf.path("raw"), args.n, args.seed)
        print(f"{len(tasks)} samples from {args.n} fingers; {len(CHAINS)} chains x {len(MBD_VALUES)} MBD",
              flush=True)
        rows: list[dict[str, Any]] = []
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(str(conf.source_path), str(PROJECT_ROOT))) as pool:
            for i, r in enumerate(pool.map(process_sample, tasks, chunksize=4), start=1):
                rows.extend(r)
                if i % 40 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} samples, {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows).sort_values(["sample", "chain", "mbd"], kind="stable")
        df.to_csv(csv_path, index=False)
        elapsed = time.time() - t0
    else:
        df = pd.read_csv(csv_path)
        elapsed = None

    configured = configured_policy(conf)
    s = summarize(df, policies_for(configured), minutiae_min=minutiae_min, nfiq2_min=nfiq2_min,
                  best_prefix=configured[:2])
    meta = {"n": args.n, "seed": args.seed, "config": args.config, "config_set_id": conf.config_set_id,
            "minutiae_min": minutiae_min, "nfiq2_min": nfiq2_min}
    if elapsed is not None:
        meta["elapsed_s"] = round(elapsed)
    findings_path = out / "findings.md"
    findings = [ln[2:].strip() for ln in findings_path.read_text().splitlines() if ln.startswith("- ")] \
        if findings_path.exists() else []
    (out / "summary.md").write_text(render_summary(s, meta=meta, findings=findings), encoding="utf-8")
    print(f"wrote {out / 'summary.md'}" + ("" if args.summarize_only else f" and {csv_path}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

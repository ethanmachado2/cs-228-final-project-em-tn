"""Telemetry -> one feature row per controller run, by threat tier (spec a2a-agents-telemetry, Req 8).

A run is one controller `SendMessage` (server side). Its stage hops are the stage agents' server
records whose `parent_task_id` is the controller task and whose start lies inside the controller
hop (the controller is serial, so continued presentations of one task do not mix). Warm-up hops
are dropped (Req 8.3). Column names carry their group as a prefix, so tier masks are exact:

- `t1__`     sizes and timing (what a passive observer of encrypted traffic sees)
- `t2__`     structure and lifecycle (skills, counts, bigrams, task states, errors)
- `t3__`     payload content (NFIQ 2 scores, minutiae counts, enhanced-image PNG size)
- `oracle__` the match score and decision; excluded from the primary features
- `cov__`    covariates outside every tier: input bytes, width, height (Req 8.2)

`export` joins labels on `trial_id` only at the end (Req 8.3). `negative-control` fits an L1
logistic regression per tier with subject-grouped 5-fold CV AUC, for C vs B and for arm A split
randomly into two labels (expected AUC ~0.5, Req 8.5).

    python -m fpv.telemetry_features export --out data/labels/features.csv
    python -m fpv.telemetry_features negative-control data/labels/features.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

FEATURE_SET_VERSION = "fs1"
STAGE_ROLES = ("quality", "preprocess", "extract", "adapter", "matcher")
SKILLS = ("assess_quality", "preprocess", "quality_after", "extract", "adapt", "match")
MAX_ATTEMPTS = 3
TIERS = ("t1", "t2", "t3", "oracle")
STATES = ("TASK_STATE_COMPLETED", "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_REJECTED", "TASK_STATE_FAILED")
LABEL_COLUMNS = ("arm", "genuine", "alteration", "difficulty", "subject", "split", "label_warmup")


def read_hops(telemetry: Path) -> list[dict[str, Any]]:
    hops = []
    for path in sorted(Path(telemetry).glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("kind") == "hop" and rec.get("side") == "server" and rec.get("rpc_method") == "SendMessage":
                    hops.append(rec)
    return hops


def _result_data(hop: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(data part, raw parts) of the response: the artifact when completed, else the status message."""
    task = ((hop.get("resp_body") or {}).get("result") or {}).get("task") or {}
    parts = [p for a in task.get("artifacts") or [] for p in a.get("parts") or []]
    parts = parts or (((task.get("status") or {}).get("message") or {}).get("parts") or [])
    data = next((p["data"] for p in parts if isinstance(p.get("data"), dict)), {})
    return data, [p for p in parts if "raw" in p]


def _request_raw(hop: dict[str, Any]) -> list[dict[str, Any]]:
    msg = ((hop.get("req_body") or {}).get("params") or {}).get("message") or {}
    return [p for p in msg.get("parts") or [] if "raw" in p]


def _size(blobs: Path, part: dict[str, Any]) -> tuple[int | None, int | None]:
    """(height, width) of an image blob, or Nones."""
    ref = part.get("raw")
    if not isinstance(ref, dict) or not (blobs / ref["$blob"]).exists():
        return None, None
    from fpv.imaging import InputImageError, decode_gray8

    try:
        img = decode_gray8((blobs / ref["$blob"]).read_bytes())
    except InputImageError:
        return None, None
    return int(img.shape[0]), int(img.shape[1])


def _blob_len(part: dict[str, Any]) -> int | None:
    ref = part.get("raw")
    return ref.get("len") if isinstance(ref, dict) else None


def run_features(ctrl: dict[str, Any], stage_hops: Sequence[dict[str, Any]], blobs: Path) -> dict[str, Any]:
    outcome, _ = _result_data(ctrl)
    hops = sorted(stage_hops, key=lambda h: h["t_start_ns"])
    seq = [h.get("skill") for h in hops]
    row: dict[str, Any] = {
        "run_id": outcome.get("run_id"), "trial_id": ctrl.get("trial_id"), "task_id": ctrl.get("task_id"),
        "control_mode": ctrl.get("control_mode") or "controller", "operation": ctrl.get("skill"),
        "t_wall": ctrl.get("t_wall"), "feature_set_version": FEATURE_SET_VERSION,
    }
    # T1: sizes and timing
    row["t1__run_latency_ms"] = ctrl["latency_ms"]
    row["t1__ctrl_req_bytes"] = ctrl["req_bytes"]
    row["t1__ctrl_resp_bytes"] = ctrl["resp_bytes"]
    row["t1__stage_req_bytes"] = sum(h["req_bytes"] for h in hops)
    row["t1__stage_resp_bytes"] = sum(h["resp_bytes"] for h in hops)
    row["t1__stage_latency_ms"] = sum(h["latency_ms"] for h in hops)
    gaps = [b["t_start_ns"] - a["t_end_ns"] for a, b in zip(hops, hops[1:])]
    row["t1__gap_ms_mean"] = sum(gaps) / len(gaps) / 1e6 if gaps else 0.0
    for s in SKILLS:
        hs = [h for h in hops if h.get("skill") == s]
        row[f"t1__{s}_latency_ms_sum"] = sum(h["latency_ms"] for h in hs)
        row[f"t1__{s}_latency_ms_max"] = max((h["latency_ms"] for h in hs), default=0.0)
        row[f"t1__{s}_req_bytes_sum"] = sum(h["req_bytes"] for h in hs)
        row[f"t1__{s}_resp_bytes_sum"] = sum(h["resp_bytes"] for h in hs)
    # T2: structure and lifecycle
    row["t2__hops"] = len(hops)
    for s in SKILLS:
        row[f"t2__n_{s}"] = seq.count(s)
    for a in SKILLS:
        for b in SKILLS:
            row[f"t2__bigram_{a}__{b}"] = sum(1 for x, y in zip(seq, seq[1:]) if (x, y) == (a, b))
    for st in STATES:
        row[f"t2__ctrl_{st.removeprefix('TASK_STATE_').lower()}"] = int(ctrl.get("final_state") == st)
    row["t2__failed_hops"] = sum(h.get("final_state") == "TASK_STATE_FAILED" for h in hops)
    row["t2__rejected_hops"] = sum(h.get("final_state") == "TASK_STATE_REJECTED" for h in hops)
    row["t2__presentation_no"] = outcome.get("presentation_no")
    # T3: visible content
    qb = [_result_data(h)[0].get("score") for h in hops if h.get("skill") == "assess_quality"]
    row["t3__quality_before"] = qb[0] if qb else None
    by_skill: dict[str, list[dict[str, Any]]] = {s: [h for h in hops if h.get("skill") == s] for s in SKILLS}
    for k in range(MAX_ATTEMPTS):
        qa = by_skill["quality_after"][k] if k < len(by_skill["quality_after"]) else None
        ex = by_skill["extract"][k] if k < len(by_skill["extract"]) else None
        ad = by_skill["adapt"][k] if k < len(by_skill["adapt"]) else None
        pre = by_skill["preprocess"][k] if k < len(by_skill["preprocess"]) else None
        row[f"t3__quality_after_{k + 1}"] = _result_data(qa)[0].get("score") if qa else None
        row[f"t3__raw_minutiae_{k + 1}"] = _result_data(ex)[0].get("count") if ex else None
        row[f"t3__minutiae_{k + 1}"] = _result_data(ad)[0].get("template_count") if ad else None
        enhanced = [p for p in _result_data(pre)[1] if p.get("filename") == "enhanced"] if pre else []
        row[f"t3__enhanced_png_bytes_{k + 1}"] = _blob_len(enhanced[0]) if enhanced else None
    # oracle
    row["oracle__score"] = outcome.get("score")
    row["oracle__status"] = outcome.get("status")
    row["oracle__match"] = None if outcome.get("status") is None else int(outcome.get("status") == "MATCH")
    # covariates
    sample = [p for p in _request_raw(ctrl) if not p.get("filename")]
    row["cov__input_bytes"] = _blob_len(sample[0]) if sample else None
    row["cov__height"], row["cov__width"] = _size(blobs, sample[0]) if sample else (None, None)
    return row


def reduce(telemetry: Path) -> list[dict[str, Any]]:
    """One row per non-warm-up controller run, in time order (Req 8.1, 8.3, 8.4)."""
    hops = read_hops(telemetry)
    blobs = Path(telemetry) / "blobs"
    stage = [h for h in hops if h.get("agent") in STAGE_ROLES and not h.get("warmup")]
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for h in stage:
        by_parent.setdefault(h.get("parent_task_id") or "", []).append(h)
    rows = []
    for c in sorted((h for h in hops if h.get("agent") == "controller" and not h.get("warmup")),
                    key=lambda h: h["t_start_ns"]):
        inside = [h for h in by_parent.get(c.get("task_id") or "", [])
                  if c["t_start_ns"] <= h["t_start_ns"] <= c["t_end_ns"]]
        rows.append(run_features(c, inside, blobs))
    return rows


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["trial_id"]] = r
    return out


def join_labels(rows: Iterable[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of labeled, non-warm-up trials, with the label columns appended."""
    out = []
    for r in rows:
        lab = labels.get(r.get("trial_id") or "")
        if lab is None or lab.get("arm") == "enroll" or lab.get("warmup"):
            continue
        out.append({**r, **{k: lab.get("warmup" if k == "label_warmup" else k) for k in LABEL_COLUMNS}})
    return out


def tier_columns(columns: Iterable[str], tier: str) -> list[str]:
    return [c for c in columns if c.startswith(f"{tier}__") and c != "oracle__status"]


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- negative control


def _num(v: Any) -> float:
    if v in (None, ""):
        return math.nan
    return float(v)


def cv_auc(rows: Sequence[dict[str, Any]], y: Sequence[int], cols: Sequence[str], seed: int = 0) -> float:
    """Subject-grouped stratified 5-fold CV ROC AUC of an L1 logistic regression on `cols`."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[_num(r[c]) for c in cols] for r in rows], dtype=float)
    X[:, np.all(np.isnan(X), axis=0)] = 0.0  # all-missing columns carry nothing
    y_arr, groups = np.asarray(y), np.array([str(r.get("subject")) for r in rows])
    folds = min(5, int(np.bincount(y_arr).min()))
    if folds < 2:
        return math.nan
    scores = np.zeros(len(rows))
    for train, test in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y_arr, groups):
        if len(set(y_arr[train])) < 2:
            return math.nan
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, random_state=seed))
        model.fit(X[train], y_arr[train])
        scores[test] = model.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y_arr, scores))


def negative_control(rows: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, dict[str, float]]:
    """AUC per tier for C vs B, and for arm A with random labels (Req 8.5)."""
    import numpy as np

    cols = list(rows[0]) if rows else []
    tiers = {t: tier_columns(cols, t) for t in TIERS}
    tiers["t1+t2"] = tiers["t1"] + tiers["t2"]
    cb = [r for r in rows if r["arm"] in ("C", "B")]
    a = [r for r in rows if r["arm"] == "A"]
    y_rand = np.random.default_rng(seed).permutation(np.arange(len(a)) % 2)
    return {
        "C vs B": {t: cv_auc(cb, [int(r["arm"] == "C") for r in cb], c, seed) for t, c in tiers.items()},
        "A random split": {t: cv_auc(a, list(y_rand), c, seed) for t, c in tiers.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpv.telemetry_features")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="reduce telemetry, join labels, write CSV")
    e.add_argument("--telemetry", type=Path, default=Path("data/telemetry"))
    e.add_argument("--labels", type=Path, default=Path("data/labels/labels.jsonl"))
    e.add_argument("--out", type=Path, default=Path("data/labels/features.csv"))
    n = sub.add_parser("negative-control", help="AUC per tier for C vs B and for randomly split arm A")
    n.add_argument("features", type=Path)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--control-mode", default="controller")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        rows = join_labels(reduce(args.telemetry), read_labels(args.labels))
        write_csv(rows, args.out)
        print(f"wrote {len(rows)} rows to {args.out}")
        return 0
    with open(args.features, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["control_mode"] == args.control_mode]
    print(f"{len(rows)} rows ({args.control_mode}): " +
          ", ".join(f"{a} {sum(r['arm'] == a for r in rows)}" for a in ("A", "B", "C", "C'")))
    for contrast, aucs in negative_control(rows, args.seed).items():
        print(f"{contrast:16} " + "  ".join(f"{t} {v:.3f}" for t, v in aucs.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

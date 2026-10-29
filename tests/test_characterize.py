"""Unit tests for the characterization summary aggregation over a tiny fake results table."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "characterize.py"


@pytest.fixture(scope="module")
def ch():
    spec = importlib.util.spec_from_file_location("characterize", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["characterize"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _rows(sample, level, raw_q3, raw_q2, counts):
    """counts: {(chain, mbd): dedup_count}"""
    return [dict(sample=sample, finger_key=sample.split("_")[0], difficulty=level, alteration="x",
                 raw_q3=raw_q3, raw_q2=raw_q2, chain=c, mbd=m, qa3=40.0 if n else None,
                 qa2=20.0 if n else None, configured_attempt=int(c[1]), dedup_count=n,
                 raw_count=n, t_preprocess=1.0, t_qa3=2.0, t_segmentation=0.5, t_orientation=0.1,
                 t_frequency=0.1, t_enhancement=0.3, t_extract=0.2, error=None)
            for (c, m), n in counts.items()]


@pytest.fixture
def df():
    # Sample a: A1 suffices. b: only A2. c: only A3. d: never (and NFIQ 2 unscorable).
    rows = []
    rows += _rows("1_a", "Real", 30, 20, {("A1", 10): 15, ("A2", 10): 0, ("A3", 6): 0})
    rows += _rows("1_b", "Easy", 8, None, {("A1", 10): 3, ("A2", 10): 12, ("A3", 6): 20})
    rows += _rows("2_c", "Real", 4, 2, {("A1", 10): 0, ("A2", 10): 11, ("A3", 6): 13})
    rows += _rows("2_d", "Hard", None, None, {("A1", 10): 0, ("A2", 10): 0, ("A3", 6): 9})
    return pd.DataFrame(rows)


def test_policy_per_attempt_and_cumulative(ch, df):
    t = ch.policy_table(df, [("A1", 10), ("A2", 10), ("A3", 6)], minutiae_min=12)
    allr = t[t["difficulty"] == "All"].set_index("attempt")
    assert list(allr["attempt_pass"]) == [0.25, 0.25, 0.5]
    assert list(allr["cumulative"]) == [0.25, 0.5, 0.75]
    assert list(allr["newly_acquired"]) == [1, 1, 1]
    real = t[t["difficulty"] == "Real"].set_index("attempt")
    assert list(real["cumulative"]) == [0.5, 0.5, 1.0]


def test_policy_end_to_end_applies_raw_gate(ch, df):
    t = ch.policy_table(df, [("A1", 10), ("A2", 10), ("A3", 6)], minutiae_min=12, nfiq2_min=5)
    allr = t[t["difficulty"] == "All"].set_index("attempt")
    # c (raw 4) and d (unscorable) fail the gate, so only a and b are ever acquired.
    assert list(allr["cumulative"]) == [0.25, 0.5, 0.5]


def test_nfiq2_table_pass_rates_count_unscorable_as_fail(ch, df):
    t = ch.nfiq2_table(df, cutoffs=(0, 5, 10))
    r3 = t[(t["upscale"] == "3x") & (t["difficulty"] == "All")].iloc[0]
    assert r3["n"] == 4 and r3["unscorable"] == 0.25
    assert (r3["pass>=0"], r3["pass>=5"], r3["pass>=10"]) == (0.75, 0.5, 0.25)
    assert r3["median"] == 8
    r2 = t[(t["upscale"] == "2x") & (t["difficulty"] == "All")].iloc[0]
    assert r2["unscorable"] == 0.5 and r2["pass>=5"] == 0.25


def test_variant_table_and_best_third(ch, df):
    v = ch.variant_table(df, minutiae_mins=(10, 12))
    a3 = v[(v["chain"] == "A3") & (v["difficulty"] == "All")].iloc[0]
    assert a3["pass>=12"] == 0.5 and a3["pass>=10"] == 0.5 and a3["count_median"] == 11
    best = ch.best_next_attempt(df, [("A1", 10), ("A2", 10)], minutiae_min=12)
    assert (best.iloc[0]["chain"], best.iloc[0]["mbd"]) == ("A3", 6)
    assert best.iloc[0]["All"] == 0.75


def test_render_summary_contains_sections(ch, df):
    s = ch.summarize(df, {"configured": [("A1", 10), ("A2", 10), ("A3", 6)]}, minutiae_min=12,
                     nfiq2_min=5, best_prefix=[("A1", 10), ("A2", 10)])
    assert s.n_fingers == 2 and s.n_samples == 4 and s.errors == 0
    md = ch.render_summary(s, meta={"n": 2, "seed": 0}, findings=["example finding"])
    for heading in ("Raw NFIQ 2", "quality_after of the configured attempts", "Policy: configured",
                    "Policy: configured (end-to-end)", "Best third attempt", "stricter floor",
                    "Median stage timings", "example finding"):
        assert heading in md
    assert "75.0%" in md and not any(math.isnan(x) for x in s.timings["t_extract"])


def test_dedup_count_unscales_and_merges(ch):
    from collections import namedtuple

    M = namedtuple("M", "x y direction quality", defaults=(0.0, 0.5))
    # scale 2, half-up: (0,0) (1,1) (2,2) (3,3) and (300,0) -> (150,0) clipped to (9,0) and
    # (299,1) -> (150,1) clipped to (9,1); (18,0) -> (9,0) merges with the clipped point.
    pts = [M(0, 0), M(1, 1), M(4, 4), M(5, 5), M(300, 0), M(18, 0)]
    assert ch.dedup_count(pts, 2.0, (10, 10)) == 5
    assert ch.dedup_count([M(0, 0), M(1, 0)], 3.0, (10, 10)) == 1


def test_quality_after_table_compares_3x_and_2x(ch, df):
    t = ch.quality_after_table(df, cutoffs=(30,))
    a1 = t[(t["chain"] == "A1") & (t["difficulty"] == "All")].set_index("upscale")
    # A1 counts are 15, 3, 0, 0 -> two scorable samples (fake qa is None when the count is 0).
    assert a1.loc["3x", "unscorable"] == 0.5 and a1.loc["3x", "median"] == 40
    assert a1.loc["2x", "median"] == 20 and a1.loc["3x", ">=30"] == 0.5 and a1.loc["2x", ">=30"] == 0

"""Unit tests for the shared attempt loop with fake stages (Req 5.3, 5.4, 6.3). No toolchain."""

from __future__ import annotations

import contextlib
import dataclasses
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fpv import config as C
from fpv.acquire import Acquired, AcquisitionRejected, acquire_template, adapt
from fpv.quality import QualityToolError
from fpv.types import Status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAPE = (103, 96)
M = namedtuple("M", "x y direction type quality", defaults=(0.0, "ending", 0.5))
IMAGE = np.zeros(SHAPE, np.uint8)


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT)


def distinct(n: int, s: float = 3.0) -> list[M]:
    """n minutiae on distinct original pixels (enhanced-image coordinates at scale s)."""
    return [M(3 * s * (i % 20) + 1, 3 * s * (i // 20) + 1) for i in range(n)]


class FakeStages:
    """Per-attempt minutiae lists; `raises[k]` makes stage `name` of attempt k raise."""

    def __init__(self, minutiae: dict[int, list], raises: dict[int, tuple[str, Exception]] | None = None):
        self.minutiae = minutiae
        self.raises = raises or {}
        self.calls: list[tuple[str, int]] = []

    def _hit(self, stage: str, k: int) -> None:
        self.calls.append((stage, k))
        if k in self.raises and self.raises[k][0] == stage:
            raise self.raises[k][1]

    def preprocess(self, image, attempt, configuration_id):
        self._hit("preprocess", attempt.attempt_no)
        return SimpleNamespace(attempt=attempt, original_shape=SHAPE, scale_factor=attempt.scale_factor,
                               configuration_id=configuration_id, timings_ms={"segmentation": 1.0})

    def quality_after(self, pre):
        self._hit("quality_after", pre.attempt.attempt_no)
        return SimpleNamespace(score=40 + pre.attempt.attempt_no, actionable_feedback={})

    def extract(self, pre):
        self._hit("extract", pre.attempt.attempt_no)
        ms = tuple(self.minutiae[pre.attempt.attempt_no])
        return SimpleNamespace(minutiae=ms, count=len(ms), configuration_id=pre.configuration_id)

    def adapt(self, ex, pre):
        self._hit("adapt", pre.attempt.attempt_no)
        return adapt(ex, pre)


def attempts_run(stages: FakeStages) -> list[int]:
    return sorted({k for s, k in stages.calls if s == "preprocess"})


def test_stops_at_first_attempt_reaching_minutiae_min(conf):
    assert conf.thresholds.minutiae_min == 12
    st = FakeStages({1: distinct(5), 2: distinct(12), 3: distinct(30)})
    res = acquire_template(IMAGE, conf, st)
    assert isinstance(res, Acquired)
    assert attempts_run(st) == [1, 2]
    assert res.attempt_no == 2 and res.template.count == 12
    assert res.configuration_id == conf.configuration_id(2)
    assert [(r.attempt_no, r.minutiae_count, r.quality_after) for r in res.attempts] == [(1, 5, 41), (2, 12, 42)]
    assert st.calls[:4] == [("preprocess", 1), ("quality_after", 1), ("extract", 1), ("adapt", 1)]


def test_gate_uses_adapted_count_after_dedup(conf):
    # 15 raw SBMEX minutiae, but at s = 3 pairs one pixel apart collapse: adapted count < 12.
    dupes = distinct(8) + [M(m.x - 1, m.y) for m in distinct(7)]
    st = FakeStages({1: dupes, 2: distinct(12), 3: []})
    res = acquire_template(IMAGE, conf, st)
    assert isinstance(res, Acquired) and res.attempt_no == 2
    first = res.attempts[0]
    assert first.raw_count == 15 and first.minutiae_count == 8


def test_rejects_after_max_attempts(conf):
    st = FakeStages({1: distinct(3), 2: distinct(11), 3: distinct(0)})
    res = acquire_template(IMAGE, conf, st)
    assert isinstance(res, AcquisitionRejected)
    assert (res.status, res.reason) == (Status.REJECTED_MINUTIAE, "insufficient_minutiae")
    assert attempts_run(st) == [1, 2, 3] and len(res.attempts) == conf.max_attempts == 3
    assert [r.configuration_id for r in res.attempts] == [conf.configuration_id(k) for k in (1, 2, 3)]


def test_respects_max_attempts_below_defined_attempts(conf):
    one = dataclasses.replace(conf, max_attempts=1)
    st = FakeStages({1: distinct(3), 2: distinct(20), 3: distinct(20)})
    res = acquire_template(IMAGE, one, st)
    assert isinstance(res, AcquisitionRejected) and attempts_run(st) == [1]


def test_failed_attempt_moves_on(conf):
    st = FakeStages({1: distinct(20), 2: distinct(20), 3: []}, raises={1: ("preprocess", RuntimeError("pyfing"))})
    res = acquire_template(IMAGE, conf, st)
    assert isinstance(res, Acquired) and res.attempt_no == 2
    assert res.attempts[0].error == "RuntimeError: pyfing" and res.attempts[0].minutiae_count is None


def test_all_attempts_raising_is_error(conf):
    boom = {k: ("extract", ValueError("bad")) for k in (1, 2, 3)}
    res = acquire_template(IMAGE, conf, FakeStages({k: [] for k in (1, 2, 3)}, raises=boom))
    assert isinstance(res, AcquisitionRejected)
    assert (res.status, res.reason) == (Status.ERROR, "preprocess_error")


def test_quality_tool_error_propagates(conf):
    st = FakeStages({1: distinct(20)}, raises={1: ("quality_after", QualityToolError("nfiq2 missing"))})
    with pytest.raises(QualityToolError):
        acquire_template(IMAGE, conf, st)


def test_span_wraps_every_stage_with_details(conf):
    seen = []

    @contextlib.contextmanager
    def span(stage, attempt_no, configuration_id):
        d: dict = {}
        yield d
        seen.append((stage, attempt_no, configuration_id, d))

    acquire_template(IMAGE, conf, FakeStages({1: distinct(14) + distinct(2)}), span=span)
    assert [s[0] for s in seen] == ["preprocess", "quality_after", "extract", "adapt"]
    assert all(s[2] == conf.configuration_id(1) for s in seen)
    assert seen[-1][3]["count"] == 14 and seen[-1][3]["dedup_removed"] == 2 and seen[-1][3]["passed"]


def test_adapter_version_must_match_config(conf):
    other = dataclasses.replace(conf, adapter_version="xyt-adapter/2")
    with pytest.raises(ValueError, match="adapter"):
        acquire_template(IMAGE, other, FakeStages({}))

"""`A2AStages` against in-process stage agents: error mapping and parity with `LocalStages` (Task 3, Req 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("a2a")

import httpx  # noqa: E402

from fpv import config as C  # noqa: E402
from fpv.adapter import XytMinutia, XytTemplate, write_xyt  # noqa: E402
from fpv.agents import codec, serve  # noqa: E402
from fpv.agents.client import A2AStages, run_metadata  # noqa: E402
from fpv.matcher import InsufficientMinutiae, MatcherError, MatchResult  # noqa: E402
from fpv.telemetry import TelemetryWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def a2a_stages(conf, tmp_path: Path, stages=None, writer=True) -> A2AStages:
    transports = {}
    for name in A2AStages.STAGE_AGENTS:
        w = TelemetryWriter(tmp_path, name) if writer else None
        app = serve.build_app(name, serve.stage_handlers(name, conf, stages), writer=w)
        transports[name] = httpx.ASGITransport(app=app)
    return A2AStages(conf, writer=TelemetryWriter(tmp_path, "controller") if writer else None, transports=transports)


class FakeMatcher:
    def __init__(self, exc=None):
        self.exc = exc

    def match(self, probe, gallery):
        if self.exc:
            raise self.exc
        return MatchResult(7, 12, 12, "fake", 0.5)


def _tpl(n: int) -> XytTemplate:
    return XytTemplate(tuple(XytMinutia(i, i, i, 50) for i in range(n)))


@pytest.fixture(scope="module")
def conf():
    return C.load(ROOT / "configs" / "pipeline.v1.yaml", root=ROOT)


@pytest.mark.parametrize("exc, expected", [(InsufficientMinutiae("probe", 3), InsufficientMinutiae),
                                           (MatcherError("bozorth3 timed out"), MatcherError),
                                           (RuntimeError("disk"), codec.RemoteStageError)])
def test_remote_errors_raise_phase1_exceptions(tmp_path, conf, exc, expected):
    for n in ("p", "g"):
        write_xyt(_tpl(12), tmp_path / f"{n}.xyt")
    st = a2a_stages(conf, tmp_path, FakeMatcher(exc))
    try:
        with pytest.raises(expected) as info:
            st.match(tmp_path / "p.xyt", tmp_path / "g.xyt")
        assert str(info.value) == str(exc)
    finally:
        st.close()


def test_run_metadata_reaches_every_stage_hop(tmp_path, conf):
    for n in ("p", "g"):
        write_xyt(_tpl(12), tmp_path / f"{n}.xyt")
    st = a2a_stages(conf, tmp_path, FakeMatcher())
    try:
        with run_metadata(parent_task_id="T1", trial_id="tr9", control_mode="controller"):
            assert st.match(tmp_path / "p.xyt", tmp_path / "g.xyt").score == 7
        st.match(tmp_path / "p.xyt", tmp_path / "g.xyt")
    finally:
        st.close()
    hops = [json.loads(x) for x in (tmp_path / "matcher.jsonl").read_text().splitlines()]
    posts = [h for h in hops if h["http_method"] == "POST"]
    assert [(h["parent_task_id"], h["trial_id"], h["control_mode"]) for h in posts] == \
        [("T1", "tr9", "controller"), (None, None, None)]


def test_unreachable_agent_is_a_stage_error(conf):
    st = A2AStages(conf, urls={"quality": "http://127.0.0.1:9/"})
    try:
        with pytest.raises(codec.RemoteStageError, match="quality unreachable"):
            st.assess_quality(np.zeros((10, 10), np.uint8))
    finally:
        st.close()


def _samples(n: int, seed: int = 7) -> list[Path]:
    real = sorted((ROOT / "data/raw/Real").glob("*.BMP"))
    alt = sorted((ROOT / "data/raw/Altered").glob("*/*.BMP"))
    rng = np.random.default_rng(seed)
    return [real[i] for i in rng.choice(len(real), n // 2, replace=False)] + \
           [alt[i] for i in rng.choice(len(alt), n - n // 2, replace=False)]


@pytest.mark.toolchain
@pytest.mark.dataset
def test_a2a_stages_match_local_stages_on_seeded_samples(tmp_path):
    from fpv.acquire import Acquired, acquire_template
    from fpv.imaging import load_gray8
    from fpv.pipeline import LocalStages

    conf = C.load()  # the frozen config, as the agents use
    local, remote = LocalStages(conf), a2a_stages(conf, tmp_path, writer=False)
    templates = []
    try:
        for path in _samples(20):
            img = load_gray8(path)
            ql, qr = local.assess_quality(img), remote.assess_quality(img)
            assert (ql.score, dict(ql.actionable_feedback)) == (qr.score, dict(qr.actionable_feedback)), path
            al, ar = acquire_template(img, conf, local), acquire_template(img, conf, remote)
            assert type(al) is type(ar), path
            assert al.attempts == ar.attempts, path
            assert al.template == ar.template, path
            if isinstance(al, Acquired):
                assert al.extraction.minutiae == ar.extraction.minutiae
                assert np.array_equal(al.preprocess.enhanced, ar.preprocess.enhanced)
                templates.append(al.template)
        for i, (p, g) in enumerate(zip(templates, templates[1:] + templates[:1])):
            pp, gp = tmp_path / f"p{i}.xyt", tmp_path / f"g{i}.xyt"
            write_xyt(p, pp)
            write_xyt(g, gp)
            for a, b in ((pp, gp), (pp, pp)):
                try:
                    ml = local.match(a, b)
                except InsufficientMinutiae as exc:
                    with pytest.raises(InsufficientMinutiae, match=str(exc)):
                        remote.match(a, b)
                    continue
                mr = remote.match(a, b)
                assert (ml.score, ml.probe_count, ml.gallery_count) == (mr.score, mr.probe_count, mr.gallery_count)
    finally:
        remote.close()
    assert len(templates) >= 15

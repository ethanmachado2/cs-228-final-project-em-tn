"""Bozorth3 against real templates. Run in the container (Req 9.3, 9.4, 9.11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpv import config as C
from fpv.adapter import XytMinutia, XytTemplate, write_xyt
from fpv.cli import EXIT_NO_MATCH, EXIT_OK, main
from fpv.matcher import match

pytestmark = pytest.mark.toolchain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"
RAW = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(CONFIG, root=PROJECT_ROOT)


def random_template(seed: int, n: int = 30, w: int = 96, h: int = 103) -> XytTemplate:
    """`n` minutiae at distinct random positions in a SOCOFing-sized frame."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(w * h, size=n, replace=False)
    return XytTemplate(tuple(
        XytMinutia(x=int(c % w), y=int(c // w), theta=int(t), quality=int(q))
        for c, t, q in zip(cells, rng.integers(0, 360, n), rng.integers(40, 100, n), strict=True)))


def test_self_match_high_and_unrelated_low(tmp_path, conf):
    threshold = conf.thresholds.bozorth3_match
    a, b = tmp_path / "a.xyt", tmp_path / "b.xyt"
    write_xyt(random_template(1), a)
    write_xyt(random_template(2), b)
    self_score = match(a, a).score
    cross = match(a, b).score
    assert self_score >= 3 * threshold, self_score
    assert cross < threshold, cross
    assert match(b, a).score < threshold


def test_cli_match_prints_score_and_decision(tmp_path):
    a, b = tmp_path / "a.xyt", tmp_path / "b.xyt"
    write_xyt(random_template(1), a)
    write_xyt(random_template(2), b)
    for gallery, code_expected, word in ((a, EXIT_OK, "MATCH "), (b, EXIT_NO_MATCH, "NO_MATCH")):
        lines: list[str] = []
        code = main(["--config", str(CONFIG), "match", str(a), str(gallery)],
                    out=lines.append, err=lambda s: None)
        text = "\n".join(lines)
        assert code == code_expected, text
        assert "score" in text and f"decision  {word}" in text and "bozorth3_match=40" in text


@pytest.mark.dataset
def test_full_chain_altered_variant_beats_other_subject(tmp_path, conf):
    """Tasks 3-6 then Bozorth3: Real 100 L index vs its Altered-Easy CR variant vs subject 101."""
    from fpv.acquire import Acquired, acquire_template
    from fpv.imaging import load_gray8

    samples = {
        "real": RAW / "Real" / "100__M_Left_index_finger.BMP",
        "altered": RAW / "Altered" / "Altered-Easy" / "100__M_Left_index_finger_CR.BMP",
        "other": RAW / "Real" / "101__M_Left_index_finger.BMP",
    }
    paths = {}
    for name, sample in samples.items():
        result = acquire_template(load_gray8(sample), conf)
        assert isinstance(result, Acquired), (name, result)
        paths[name] = tmp_path / f"{name}.xyt"
        write_xyt(result.template, paths[name])

    genuine = match(paths["altered"], paths["real"]).score
    impostor = match(paths["other"], paths["real"]).score
    assert genuine > impostor, (genuine, impostor)
    assert genuine >= conf.thresholds.bozorth3_match > impostor, (genuine, impostor)

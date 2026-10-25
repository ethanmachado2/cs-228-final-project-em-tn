"""`fpv extract` end to end on a real SOCOFing sample. Run in the container (Req 6.6, 7.6, 5.8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpv import config as C
from fpv.adapter import read_xyt
from fpv.cli import main

pytestmark = [pytest.mark.toolchain, pytest.mark.dataset]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_SAMPLE = PROJECT_ROOT / "data/raw/Real/100__M_Left_index_finger.BMP"


def test_fpv_extract_writes_deterministic_template(tmp_path):
    conf = C.load(PROJECT_ROOT / "configs" / "pipeline.v1.yaml", root=PROJECT_ROOT)
    outputs = []
    texts = []
    for i in range(2):
        lines: list[str] = []
        out = tmp_path / f"probe{i}.xyt"
        code = main(["--config", str(conf.source_path), "extract", str(REAL_SAMPLE), "--out", str(out)],
                    out=lines.append, err=lambda s: None)
        assert code == 0, lines
        outputs.append(read_xyt(out))
        texts.append(out.read_bytes())
        joined = "\n".join(lines)
        assert "configuration_id  cfg_" in joined and "minutiae" in joined

    t = outputs[0]
    assert texts[0] == texts[1]                                  # Req 5.8
    assert t.count >= conf.thresholds.minutiae_min
    assert all(0 <= m.x < 96 and 0 <= m.y < 103 for m in t.minutiae)   # original 96x103 frame
    assert len({(m.x, m.y) for m in t.minutiae}) == t.count
    for line in texts[0].decode().splitlines():
        assert len(line.split()) == 4 and all(p.isdigit() for p in line.split())

"""Catalog checks against the real SOCOFing data. Run: `docker compose run --rm fpv pytest -m dataset`."""

from __future__ import annotations

import numpy as np
import pytest

from fpv import cli
from fpv.catalog import Catalog
from fpv.imaging import load_gray8
from fpv.types import Finger, Hand, Identity

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def catalog():
    from conftest import PROJECT_ROOT

    data = PROJECT_ROOT / "data"
    return Catalog.load(data / "processed" / "dataset_splits.csv",
                        data / "quality_reports" / "duplicate_exclusions.csv", data / "raw")


def test_real_sample_resolves(catalog, project_root):
    path = project_root / "data/raw/Real/100__M_Left_index_finger.BMP"
    rec = catalog.record(path)
    assert rec.source_path == "Real/100__M_Left_index_finger.BMP"
    assert rec.label.identity == Identity(100, Hand.LEFT, Finger.INDEX)
    assert (rec.label.gender, rec.label.alteration, rec.label.difficulty) == ("M", "Real", "Real")
    assert (rec.split, rec.excluded) == ("train", False)
    assert (rec.width, rec.height) == (96, 103)
    img = load_gray8(path)
    assert img.dtype == np.uint8 and img.shape == (103, 96)


def test_altered_hard_sample_resolves(catalog, project_root):
    rec = catalog.record(project_root / "data/raw/Altered/Altered-Hard/100__M_Left_index_finger_CR.BMP")
    assert rec.label.identity == Identity(100, Hand.LEFT, Finger.INDEX)
    assert (rec.label.alteration, rec.label.difficulty) == ("CR", "Hard")
    assert (rec.split, rec.excluded) == ("train", False)


def test_cli_catalog_demo(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    out: list[str] = []
    code = cli.main(["catalog", "data/raw/Altered/Altered-Easy/100__M_Left_index_finger_CR.BMP"],
                    out=out.append, err=lambda s: None)
    assert code == 0
    fields = dict(line.split(None, 1) for line in out)
    assert (fields["subject"], fields["hand"], fields["finger"]) == ("100", "Left", "index")
    assert (fields["alteration"], fields["difficulty"], fields["split"]) == ("CR", "Easy", "train")

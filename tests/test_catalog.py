"""Unit tests for fpv.catalog (Req 2.1-2.4) and `fpv catalog` (Req 2.7), on synthetic fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fpv import cli
from fpv.catalog import Catalog, dataset_key, make_sample_id, parse_label
from fpv.imaging import InputImageError
from fpv.types import Finger, Hand

DIRS = {"Real": "Real", "Easy": "Altered/Altered-Easy", "Medium": "Altered/Altered-Medium",
        "Hard": "Altered/Altered-Hard"}
SPLITS_HEADER = ("file,valid_label,subject_id,gender,hand,finger,alteration,fingerprint_key,"
                 "exclude_duplicate_conflict,split,standardized_file")
EXCL_HEADER = ("file,valid_label,subject_id,gender,hand,finger,alteration,fingerprint_key,"
               "exclude_duplicate_conflict")


# --------------------------------------------------------------------------- parse_label


def test_parse_real_label():
    lab = parse_label(Path("data/raw/Real/100__M_Left_index_finger.BMP"))
    assert lab is not None
    assert (lab.subject_id, lab.gender, lab.identity.hand, lab.identity.finger) == (100, "M", Hand.LEFT, Finger.INDEX)
    assert (lab.alteration, lab.difficulty, lab.is_real) == ("Real", "Real", True)


@pytest.mark.parametrize("difficulty", ["Easy", "Medium", "Hard"])
@pytest.mark.parametrize("suffix", ["Obl", "CR", "Zcut"])
def test_parse_altered_labels_across_difficulties(difficulty, suffix):
    lab = parse_label(Path(f"{DIRS[difficulty]}/7__F_Right_little_finger_{suffix}.BMP"))
    assert lab is not None
    assert (lab.subject_id, lab.gender, lab.identity.hand, lab.identity.finger) == (7, "F", Hand.RIGHT, Finger.LITTLE)
    assert (lab.alteration, lab.difficulty, lab.is_real) == (suffix, difficulty, False)


@pytest.mark.parametrize("name", [
    "100__M_Left_index_finger.bmp", "100__M_Left_index_finger.BMP", "100__M_Left_index_finger.Bmp",
])
def test_extension_is_case_insensitive(name):
    assert parse_label(Path("Real") / name) is not None


def test_case_is_normalized():
    lab = parse_label(Path("Altered/altered-hard/5__f_right_THUMB_finger_zcut.bmp"))
    assert lab is not None
    assert (lab.gender, lab.identity.hand, lab.identity.finger, lab.alteration, lab.difficulty) == (
        "F", Hand.RIGHT, Finger.THUMB, "Zcut", "Hard")


@pytest.mark.parametrize("path", [
    "Real/100__M_Left_index_finger.png",          # wrong extension
    "Real/100_M_Left_index_finger.BMP",           # single underscore
    "Real/100__X_Left_index_finger.BMP",          # bad gender
    "Real/100__M_Middle_index_finger.BMP",        # bad hand
    "Real/100__M_Left_pinky_finger.BMP",          # bad finger
    "Altered/Altered-Easy/100__M_Left_index_finger_Blur.BMP",  # unknown alteration
    "Real/0__M_Left_index_finger.BMP",            # subject out of range
    "Real/601__M_Left_index_finger.BMP",
    "Real/abc__M_Left_index_finger.BMP",
    "Real/100__M_Left_index_finger.BMP.bak",
])
def test_malformed_names_are_unlabeled(path):
    assert parse_label(Path(path)) is None


@pytest.mark.parametrize("path", [
    "Real/100__M_Left_index_finger_CR.BMP",                 # Real dir with alteration suffix
    "Altered/Altered-Easy/100__M_Left_index_finger.BMP",    # Altered dir without suffix
    "Altered/Altered-Hard/100__M_Left_index_finger.BMP",
    "Altered/Altered-Extreme/100__M_Left_index_finger_CR.BMP",  # unknown difficulty
    "elsewhere/100__M_Left_index_finger.BMP",               # unknown directory
])
def test_inconsistent_directories_are_unlabeled(path):
    assert parse_label(Path(path)) is None


def test_dataset_key_matches_csv_layout():
    for rel in ("Real/1__M_Left_thumb_finger.BMP", "Altered/Altered-Medium/1__M_Left_thumb_finger_Obl.BMP"):
        assert dataset_key(Path("/anywhere/data/raw") / rel, parse_label(Path(rel))) == rel


# --------------------------------------------------------------------------- catalog fixtures


def _write_bmp(path: Path, seed: int = 0) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(seed).integers(0, 256, size=(103, 96), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return path.read_bytes()


@pytest.fixture
def mini_dataset(tmp_path):
    """A tiny SOCOFing-shaped tree plus split and exclusion CSVs."""
    raw = tmp_path / "data" / "raw"
    files = {
        "Real/1__M_Left_index_finger.BMP": ("train", False),
        "Real/2__F_Right_thumb_finger.BMP": ("validation", False),
        "Real/3__M_Left_ring_finger.BMP": (None, True),  # duplicate conflict: excluded, no split
        "Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP": ("train", False),
        "Altered/Altered-Hard/1__M_Left_index_finger_Zcut.BMP": ("train", False),
        "Altered/Altered-Medium/2__F_Right_thumb_finger_Obl.BMP": ("validation", False),
    }
    splits_rows, excl_rows = [SPLITS_HEADER], [EXCL_HEADER]
    for i, (rel, (split, excluded)) in enumerate(files.items()):
        _write_bmp(raw / rel, seed=i)
        lab = parse_label(Path(rel))
        cols = (f"{rel},True,{lab.subject_id},{lab.gender},{lab.identity.hand},{lab.identity.finger},"
                f"{lab.alteration},key,{excluded}")
        excl_rows.append(cols)
        if split is not None:
            splits_rows.append(f"{cols},{split},{rel[:-4]}.npy")
    (raw / "Real" / "notes.txt").write_text("not a sample")
    splits_csv = tmp_path / "data" / "processed" / "dataset_splits.csv"
    excl_csv = tmp_path / "data" / "quality_reports" / "duplicate_exclusions.csv"
    for path, rows in ((splits_csv, splits_rows), (excl_csv, excl_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n")
    return {"root": tmp_path, "raw": raw, "splits": splits_csv, "exclusions": excl_csv,
            "catalog": Catalog.load(splits_csv, excl_csv, raw)}


def test_split_and_exclusion_lookup(mini_dataset):
    cat, raw = mini_dataset["catalog"], mini_dataset["raw"]
    rec = cat.record(raw / "Altered/Altered-Hard/1__M_Left_index_finger_Zcut.BMP")
    assert rec.source_path == "Altered/Altered-Hard/1__M_Left_index_finger_Zcut.BMP"
    assert (rec.split, rec.excluded, rec.label.difficulty, rec.label.alteration) == ("train", False, "Hard", "Zcut")
    assert cat.record(raw / "Real/2__F_Right_thumb_finger.BMP").split == "validation"
    excluded = cat.record(raw / "Real/3__M_Left_ring_finger.BMP")
    assert excluded.excluded is True and excluded.split is None and excluded.label is not None


def test_record_hash_dimensions_and_sample_id(mini_dataset):
    cat, raw = mini_dataset["catalog"], mini_dataset["raw"]
    path = raw / "Real/1__M_Left_index_finger.BMP"
    rec = cat.record(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert rec.sha256 == digest and (rec.width, rec.height) == (96, 103)
    expected = "s_" + hashlib.sha256(f"{digest}:{rec.source_path}".encode()).hexdigest()[:16]
    assert rec.sample_id == expected == make_sample_id(digest, rec.source_path)
    assert cat.record(path) == rec  # deterministic


def test_sample_id_differs_for_identical_bytes_at_different_paths(mini_dataset, tmp_path):
    cat, raw = mini_dataset["catalog"], mini_dataset["raw"]
    src = raw / "Real/1__M_Left_index_finger.BMP"
    copy = raw / "Real/4__M_Left_index_finger.BMP"
    copy.write_bytes(src.read_bytes())
    a, b = cat.record(src), cat.record(copy)
    assert a.sha256 == b.sha256 and a.sample_id != b.sample_id


def test_unlabeled_sample_gets_no_ground_truth(mini_dataset, tmp_path):
    cat = mini_dataset["catalog"]
    outside = tmp_path / "uploads" / "my_finger.bmp"
    _write_bmp(outside)
    rec = cat.record(outside)
    assert rec.label is None and rec.split is None and rec.excluded is False
    assert rec.source_path == str(outside.resolve())  # absolute when outside the raw root


def test_record_raises_on_non_image(mini_dataset):
    with pytest.raises(InputImageError):
        mini_dataset["catalog"].record(mini_dataset["raw"] / "Real" / "notes.txt")


def test_iter_samples_filters_by_split_and_real(mini_dataset):
    cat = mini_dataset["catalog"]
    train = [r.source_path for r in cat.iter_samples("train")]
    assert train == sorted(train) and len(train) == 3
    assert [r.source_path for r in cat.iter_samples("train", real=True)] == ["Real/1__M_Left_index_finger.BMP"]
    assert {r.label.difficulty for r in cat.iter_samples("train", real=False)} == {"Easy", "Hard"}
    assert [r.label.alteration for r in cat.iter_samples("validation", real=False)] == ["Obl"]
    assert list(cat.iter_samples("test")) == []
    with pytest.raises(ValueError):
        list(cat.iter_samples("dev"))


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def mini_cli(mini_dataset, default_config_path, monkeypatch):
    raw = yaml.safe_load(default_config_path.read_text())
    raw["paths"].update(raw="data/raw", splits="data/processed/dataset_splits.csv",
                        exclusions="data/quality_reports/duplicate_exclusions.csv")
    cfg = mini_dataset["root"] / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(mini_dataset["root"])

    def run(*argv):
        out, err = [], []
        code = cli.main(["--config", str(cfg), "catalog", *argv], out=out.append, err=err.append)
        return code, "\n".join(out), "\n".join(err)

    return run


def test_cli_catalog_prints_labels_and_split(mini_cli):
    code, out, _ = mini_cli("data/raw/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP")
    assert code == 0
    fields = dict(line.split(None, 1) for line in out.splitlines())
    assert fields["subject"] == "1" and fields["hand"] == "Left" and fields["finger"] == "index"
    assert fields["alteration"] == "CR" and fields["difficulty"] == "Easy"
    assert fields["split"] == "train" and fields["excluded"] == "no"


def test_cli_catalog_rejects_missing_and_undecodable(mini_cli):
    code, _, err = mini_cli("data/raw/Real/nope.BMP")
    assert code == cli.EXIT_USAGE and "no such file" in err
    code, _, err = mini_cli("data/raw/Real/notes.txt")
    assert code == cli.EXIT_USAGE and "cannot decode" in err

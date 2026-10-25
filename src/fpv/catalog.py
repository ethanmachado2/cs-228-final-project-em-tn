"""SOCOFing dataset catalog: label parsing, split/exclusion lookup and sample records (Req 2).

Labels come from the filename and its parent directory, using the same pattern as
`data_preprocessing/validate_labels.py`. Splits come from `data/processed/dataset_splits.csv`
and duplicate-conflict exclusions from `data/quality_reports/duplicate_exclusions.csv`.
Nothing here writes to the dataset (Req 2.6).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fpv.imaging import InputImageError, decode_gray8
from fpv.types import Identity

Alteration = Literal["Real", "Obl", "CR", "Zcut"]
Difficulty = Literal["Real", "Easy", "Medium", "Hard"]

# Same pattern as data_preprocessing/validate_labels.py (case-insensitive, so `.BMP` matches).
LABEL_RE = re.compile(
    r"^(\d+)__(M|F)_(Left|Right)_"
    r"(little|ring|middle|index|thumb)_finger"
    r"(?:_(Obl|CR|Zcut))?\.bmp$",
    re.IGNORECASE,
)
_ALTERED_DIR_RE = re.compile(r"^Altered-(Easy|Medium|Hard)$", re.IGNORECASE)
_CANONICAL_ALTERATION: Mapping[str, Alteration] = {"obl": "Obl", "cr": "CR", "zcut": "Zcut"}
_CANONICAL_DIFFICULTY: Mapping[str, Difficulty] = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SampleLabel:
    identity: Identity
    gender: Literal["M", "F"]
    alteration: Alteration
    difficulty: Difficulty

    @property
    def subject_id(self) -> int:
        return self.identity.user_id

    @property
    def is_real(self) -> bool:
        return self.alteration == "Real"


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str  # "s_" + sha256(f"{sha256}:{source_path}")[:16]
    source_path: str  # relative to the raw root when inside it, else absolute
    sha256: str
    width: int
    height: int
    label: SampleLabel | None  # None => unlabeled (Req 2.2)
    split: str | None  # train | validation | test | None
    excluded: bool


def make_sample_id(sha256: str, source_path: str) -> str:
    """Content hash + path: SOCOFing has byte-identical files under conflicting labels."""
    return "s_" + hashlib.sha256(f"{sha256}:{source_path}".encode("utf-8")).hexdigest()[:16]


def _difficulty_for_dir(dirname: str) -> Difficulty | None:
    if dirname.lower() == "real":
        return "Real"
    m = _ALTERED_DIR_RE.match(dirname)
    return _CANONICAL_DIFFICULTY[m.group(1).lower()] if m else None


def parse_label(path: Path | str) -> SampleLabel | None:
    """Parse SOCOFing labels from the filename and parent directory, or None if unlabeled.

    Returns None for names that do not match the pattern, subject IDs outside 1..600,
    unknown parent directories, a `Real` directory with an alteration suffix, or an
    `Altered-*` directory without one (Req 2.1, 2.2).
    """
    p = Path(path)
    m = LABEL_RE.fullmatch(p.name)
    if m is None:
        return None
    difficulty = _difficulty_for_dir(p.parent.name)
    if difficulty is None:
        return None
    subject, gender, hand, finger, suffix = m.groups()
    if (difficulty == "Real") != (suffix is None):
        return None
    try:
        identity = Identity.parse(subject, hand, finger)
    except ValueError:
        return None
    alteration: Alteration = "Real" if suffix is None else _CANONICAL_ALTERATION[suffix.lower()]
    return SampleLabel(identity=identity, gender=gender.upper(), alteration=alteration, difficulty=difficulty)


def dataset_key(path: Path | str, label: SampleLabel) -> str:
    """The key used by the split/exclusion CSVs: `Real/<name>` or `Altered/Altered-<D>/<name>`."""
    name = Path(path).name
    if label.difficulty == "Real":
        return f"Real/{name}"
    return f"Altered/Altered-{label.difficulty}/{name}"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in required if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
            return list(reader)
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {path}: {exc.strerror or exc}") from exc


class Catalog:
    """Split and exclusion lookup over the SOCOFing tree rooted at `raw_root`."""

    def __init__(self, splits: Mapping[str, str], exclusions: Mapping[str, bool], raw_root: Path,
                 split_files: Mapping[str, tuple[str, ...]]) -> None:
        # Keys are lower-cased so lookups match the extension case-insensitively.
        self._splits = dict(splits)
        self._exclusions = dict(exclusions)
        self._split_files = dict(split_files)
        self.raw_root = Path(raw_root).resolve()

    @classmethod
    def load(cls, splits_csv: Path, exclusions_csv: Path, raw_root: Path) -> "Catalog":
        splits: dict[str, str] = {}
        split_files: dict[str, list[str]] = {}
        for row in _read_csv(Path(splits_csv), ("file", "split")):
            split = row["split"].strip()
            if not split:
                continue
            key = row["file"].strip()
            splits[key.lower()] = split
            split_files.setdefault(split, []).append(key)
        exclusions = {
            row["file"].strip().lower(): _truthy(row["exclude_duplicate_conflict"])
            for row in _read_csv(Path(exclusions_csv), ("file", "exclude_duplicate_conflict"))
        }
        return cls(splits, exclusions, raw_root, {s: tuple(sorted(f)) for s, f in split_files.items()})

    def split_of(self, key: str) -> str | None:
        return self._splits.get(key.lower())

    def is_excluded(self, key: str) -> bool:
        return self._exclusions.get(key.lower(), False)

    def source_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.raw_root).as_posix()
        except ValueError:
            return str(resolved)

    def record(self, path: Path | str) -> SampleRecord:
        """Hash, measure and label one sample; raises InputImageError if it cannot be decoded."""
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise InputImageError(f"cannot read {p}: {exc.strerror or exc}") from exc
        img = decode_gray8(data, source=str(p))
        digest = hashlib.sha256(data).hexdigest()
        source = self.source_path(p)
        label = parse_label(p)
        split, excluded = None, False
        if label is not None:  # unlabeled samples get no ground truth (Req 2.2)
            key = dataset_key(p, label)
            split, excluded = self.split_of(key), self.is_excluded(key)
        return SampleRecord(
            sample_id=make_sample_id(digest, source),
            source_path=source,
            sha256=digest,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            label=label,
            split=split,
            excluded=excluded,
        )

    def iter_labeled(self, split: str, *, real: bool | None = None) -> Iterator[tuple[str, SampleLabel]]:
        """`(dataset key, label)` of the labeled, non-excluded samples of `split`, in sorted key order.

        Reads only the split/exclusion tables (no file I/O), so trial generation stays pure.
        `real=True` yields only Real samples, `real=False` only Altered ones, `None` both.
        """
        if split not in SPLITS:
            raise ValueError(f"invalid split {split!r}; allowed values: {', '.join(SPLITS)}")
        for key in self._split_files.get(split, ()):
            label = parse_label(key)
            if label is None or (real is not None and label.is_real != real):
                continue
            if self.is_excluded(key):
                continue
            yield key, label

    def iter_samples(self, split: str, *, real: bool | None = None) -> Iterator[SampleRecord]:
        """Labeled, non-excluded samples of `split` in sorted path order.

        `real=True` yields only Real samples, `real=False` only Altered ones, `None` both.
        """
        for key, _label in self.iter_labeled(split, real=real):
            rec = self.record(self.raw_root / key)
            if rec.excluded:
                continue
            yield rec

"""Shared fixtures and auto-skip rules for the `toolchain` and `dataset` markers."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"

TOOLCHAIN_BINARIES = ("nfiq2", "bozorth3")
TOOLCHAIN_MODULES = ("pyfing", "cv2", "keras")
DATASET_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "Real",
    PROJECT_ROOT / "data" / "raw" / "Altered",
    PROJECT_ROOT / "data" / "processed" / "dataset_splits.csv",
)


def _missing_toolchain() -> list[str]:
    missing = [b for b in TOOLCHAIN_BINARIES if shutil.which(b) is None]
    missing += [m for m in TOOLCHAIN_MODULES if importlib.util.find_spec(m) is None]
    return missing


def _missing_dataset() -> list[str]:
    return [str(p.relative_to(PROJECT_ROOT)) for p in DATASET_PATHS if not p.exists()]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing_tc = _missing_toolchain()
    missing_ds = _missing_dataset()
    skip_tc = pytest.mark.skip(reason=f"toolchain unavailable (missing: {', '.join(missing_tc)})")
    skip_ds = pytest.mark.skip(reason=f"dataset unavailable (missing: {', '.join(missing_ds)})")
    for item in items:
        if missing_tc and "toolchain" in item.keywords:
            item.add_marker(skip_tc)
        if missing_ds and "dataset" in item.keywords:
            item.add_marker(skip_ds)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def default_config_path() -> Path:
    return DEFAULT_CONFIG

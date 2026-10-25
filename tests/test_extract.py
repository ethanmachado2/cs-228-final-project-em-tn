"""Unit tests for the extraction stage seams (no pyfing needed) (Req 6.1, 6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fpv import extract as X


@dataclass
class FakeParams:
    """Stand-in with the same shape as pyfing.SbmexParameters (a dataclass with defaults)."""

    binarization_threshold: int = 0
    minimum_background_distance: int = 10
    minimum_tracking_distance: int = 5
    detection_method: str = "Merge"


class FakeSbmex:
    def __init__(self, parameters: FakeParams) -> None:
        self.parameters = parameters


def test_overrides_are_merged_over_defaults():
    alg = X.build_extractor({"minimum_background_distance": 6}, parameters_cls=FakeParams,
                            algorithm_cls=FakeSbmex)
    assert alg.parameters == FakeParams(minimum_background_distance=6)
    assert X.sbmex_parameters({}, parameters_cls=FakeParams) == FakeParams()


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown SBMEX parameter"):
        X.sbmex_parameters({"min_background_distance": 6}, parameters_cls=FakeParams)


def test_extractor_cache_is_keyed_by_overrides():
    kw = dict(parameters_cls=FakeParams, algorithm_cls=FakeSbmex)
    a = X.extractor_for({"minimum_background_distance": 6, "detection_method": "Ridges"}, **kw)
    b = X.extractor_for({"detection_method": "Ridges", "minimum_background_distance": 6}, **kw)
    c = X.extractor_for({"minimum_background_distance": 8}, **kw)
    assert a is b and a is not c
    assert c.parameters.minimum_background_distance == 8


def test_encode_neighborhood_matches_pyfing_bit_layout():
    # Weights clockwise from top-left: 1 2 4 / 128 . 8 / 64 32 16.
    sk = np.zeros((5, 7), np.uint8)
    sk[2, 1:6] = 255                       # horizontal line
    enc = X.encode_neighborhood(sk)
    assert enc[2, 3] == 128 + 8            # left + right neighbors
    assert enc[2, 1] == 8 and enc[2, 5] == 128
    assert enc[1, 3] == 64 + 32 + 16       # pixel above sees the three below it
    one = np.zeros((3, 3), np.uint8)
    one[1, 1] = 1
    assert X.encode_neighborhood(np.full((3, 3), 255, np.uint8))[1, 1] == 255
    assert X.encode_neighborhood(one)[0, 0] == 16  # border pixels see zero padding


def test_extract_module_never_references_leader():
    src = Path(X.__file__).read_text(encoding="utf-8")
    assert "Leader" not in src and "minutiae_end_to_end" not in src  # Req 6.2

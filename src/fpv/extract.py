"""Minutiae extraction stage: pyfing SBMEX on an attempt's enhanced image and mask (Req 6.1, 6.2).

SBMEX runs on `PreprocessResult.enhanced` (white ridges on black) and `.mask`, so coordinates
are in pixels of the possibly upscaled image; the adapter (Task 6) undoes `scale_factor`.
The attempt's `sbmex_overrides` are merged over the `SbmexParameters` defaults, and one SBMEX
instance is cached per process for each distinct overrides key.

pyfing's end-to-end LEADER extractor is never used (Req 6.2): this module only references
`pyfing.Sbmex` / `pyfing.SbmexParameters`.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fpv.preprocess import PreprocessResult

__all__ = [
    "SBMEX_DPI",
    "ExtractionResult",
    "extract",
    "overrides_key",
    "sbmex_parameters",
    "build_extractor",
    "extractor_for",
]

# SBMEX parameters are pixel-based and pyfing ignores `dpi` in Sbmex.run; 500 is passed for clarity.
SBMEX_DPI = 500

OverridesKey = tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    minutiae: tuple[Any, ...]      # pyfing.Minutia(x, y, direction, type, quality), enhanced-image pixels
    count: int                     # raw SBMEX count (before the adapter's dedup)
    configuration_id: str
    duration_ms: float


# --------------------------------------------------------------------------- parameters + cache


def overrides_key(overrides: Mapping[str, Any]) -> OverridesKey:
    """Order-independent, hashable key for a set of SBMEX overrides."""
    return tuple(sorted(overrides.items()))


# The 8-neighborhood weights SBMEX uses to encode each skeleton pixel as one byte.
CN_FILTER = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]], dtype=np.uint8)


def encode_neighborhood(skeleton: np.ndarray, weights: np.ndarray = CN_FILTER) -> np.ndarray:
    """Byte code of each pixel's 8-neighborhood: sum of weights[i, j] * (skeleton[y+i-1, x+j-1] != 0).

    Pure-numpy replacement for `Sbmex._encode_neighborhood`. pyfing 0.7.3 computes this with
    `cv.filter2D(uint8, -1, <uint8 kernel>)`; under OpenCV 5.0.0 an integer kernel is applied
    with the wrong taps (a horizontal line encodes as 64+4 instead of 128+8), so ridge
    following stops immediately and SBMEX returns no minutiae at all (Task 5 finding). This is
    the correlation pyfing intends (what OpenCV 4 produced), exact and independent of OpenCV.
    """
    on = np.pad((skeleton != 0).astype(np.uint16), 1)
    h, w = skeleton.shape
    out = np.zeros((h, w), dtype=np.uint16)
    for i in range(3):
        for j in range(3):
            if weights[i, j]:
                out += int(weights[i, j]) * on[i:i + h, j:j + w]
    return out.astype(np.uint8)


_classes: tuple[type, type] | None = None


def _pyfing_classes() -> tuple[type, type]:
    """(SbmexParameters, Sbmex subclass with the OpenCV 5 neighborhood-encoding fix)."""
    global _classes
    if _classes is None:
        import pyfing

        class FixedSbmex(pyfing.Sbmex):
            def _encode_neighborhood(self, skeleton: np.ndarray) -> np.ndarray:  # noqa: D401
                return encode_neighborhood(skeleton, self._cn_filter)

        _classes = (pyfing.SbmexParameters, FixedSbmex)
    return _classes


def sbmex_parameters(overrides: Mapping[str, Any], *, parameters_cls: type | None = None) -> Any:
    """`SbmexParameters(**{**defaults, **overrides})`; unknown override names raise ValueError."""
    if parameters_cls is None:
        parameters_cls = _pyfing_classes()[0]
    defaults = {f.name: getattr(parameters_cls(), f.name) for f in dataclasses.fields(parameters_cls)}
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"unknown SBMEX parameter(s) {unknown}; allowed: {', '.join(sorted(defaults))}")
    return parameters_cls(**{**defaults, **dict(overrides)})


def build_extractor(overrides: Mapping[str, Any], *, parameters_cls: type | None = None,
                    algorithm_cls: type | None = None) -> Any:
    """Construct an SBMEX instance for `overrides` (the seam used by unit tests)."""
    if parameters_cls is None or algorithm_cls is None:
        p_cls, a_cls = _pyfing_classes()
        parameters_cls = parameters_cls or p_cls
        algorithm_cls = algorithm_cls or a_cls
    return algorithm_cls(sbmex_parameters(overrides, parameters_cls=parameters_cls))


_lock = threading.Lock()
_extractors: dict[tuple[Any, Any, OverridesKey], Any] = {}


def extractor_for(overrides: Mapping[str, Any], *, parameters_cls: type | None = None,
                  algorithm_cls: type | None = None) -> Any:
    """Per-process cached SBMEX instance for this overrides key."""
    key = (parameters_cls, algorithm_cls, overrides_key(overrides))
    with _lock:
        alg = _extractors.get(key)
        if alg is None:
            alg = build_extractor(overrides, parameters_cls=parameters_cls, algorithm_cls=algorithm_cls)
            _extractors[key] = alg
        return alg


# --------------------------------------------------------------------------- stage


def _plain(m: Any) -> Any:
    """Same Minutia type with built-in Python scalars (pyfing can leave numpy floats in quality)."""
    return m._replace(x=int(m.x), y=int(m.y), direction=float(m.direction), type=str(m.type),
                      quality=float(m.quality))


def extract(pre: PreprocessResult) -> ExtractionResult:
    """SBMEX on the attempt's enhanced image and mask with its overrides applied (Req 6.1)."""
    enhanced, mask = pre.enhanced, pre.mask
    for name, arr in (("enhanced", enhanced), ("mask", mask)):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError(f"{name} must be a 2-D uint8 array")
    if enhanced.shape != mask.shape:
        raise ValueError(f"enhanced {enhanced.shape} and mask {mask.shape} shapes differ")

    alg = extractor_for(pre.attempt.sbmex_overrides)
    start = time.perf_counter_ns()
    minutiae = alg.run(enhanced, mask, SBMEX_DPI)
    duration_ms = (time.perf_counter_ns() - start) / 1e6
    out = tuple(_plain(m) for m in minutiae)
    return ExtractionResult(minutiae=out, count=len(out), configuration_id=pre.configuration_id,
                            duration_ms=duration_ms)

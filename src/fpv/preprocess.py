"""Preprocessing stage: the pyfing segmentation -> orientation -> frequency -> enhancement chain.

Every attempt starts from the raw sample (Req 5.3): a bicubic `attempt.resample` (the result is
treated as a native 500 ppi image), optional CLAHE, then a bicubic upscale by `attempt.upscale`
with pyfing run at `dpi = 500 * upscale`. `scale_factor = resample * upscale`. pyfing algorithm objects are
cached per process so Keras weights load once. TensorFlow runs CPU-only with op determinism
and a fixed seed, so repeated calls give byte-identical outputs (Req 5.8).

`quality_after` scores the inverted, masked enhanced image after resizing it back to the
original 500 ppi size, with the same `QualitySettings` (and so the same NFIQ 2 upscale) as the
raw `quality_before` score (Req 5.5). A `None` score is allowed (Req 5.6).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fpv import imaging
from fpv.config import AttemptConfig, ClaheConfig
from fpv.quality import QualityResult, QualitySettings, assess_quality

__all__ = [
    "AttemptConfig",
    "ClaheConfig",
    "PreprocessResult",
    "preprocess",
    "quality_after",
    "quality_after_image",
]

STEPS = ("segmentation", "orientation", "frequency", "enhancement")
RANDOM_SEED = 0


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    attempt: AttemptConfig
    enhanced: np.ndarray               # uint8, white ridges on black (pyfing convention), upscaled size
    mask: np.ndarray                   # uint8 {0, 255}, same shape as `enhanced`
    scale_factor: float                # == attempt.scale_factor (resample * upscale)
    original_shape: tuple[int, int]    # (H, W) of the raw sample
    configuration_id: str
    timings_ms: Mapping[str, float]    # segmentation / orientation / frequency / enhancement


# --------------------------------------------------------------------------- runtime + registry

_lock = threading.Lock()
_runtime_ready = False
_algorithms: dict[str, Any] = {}


def _init_runtime() -> None:
    """CPU-only, deterministic TensorFlow/Keras. Must run before pyfing (and so keras) is imported."""
    global _runtime_ready
    if _runtime_ready:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import keras
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):  # devices already initialized; CUDA_VISIBLE_DEVICES covers it
        pass
    keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()
    _runtime_ready = True


def _factories() -> dict[str, Callable[[], Any]]:
    import pyfing

    return {
        "GMFS": pyfing.Gmfs, "SUFS": pyfing.Sufs,
        "GBFOE": pyfing.Gbfoe, "SNFOE": pyfing.Snfoe,
        "XSFFE": pyfing.Xsffe, "SNFFE": pyfing.Snffe,
        "GBFEN": pyfing.Gbfen, "SNFEN": pyfing.Snfen,
    }


def _algorithm(method: str) -> Any:
    """Per-process cached pyfing algorithm instance for `method` (e.g. "SUFS")."""
    alg = _algorithms.get(method)
    if alg is None:
        _init_runtime()
        alg = _factories()[method]()
        _algorithms[method] = alg
    return alg


# --------------------------------------------------------------------------- stage


def _snfen_resampled(alg: Any, image: np.ndarray, mask: np.ndarray, orientation: np.ndarray,
                     periods: np.ndarray, dpi: int | float) -> np.ndarray:
    """SNFEN at a non-native dpi, working around a pyfing 0.7.3 bug.

    `Snfen.run(..., dpi != 500)` resizes the ridge-period map to the *original* image size
    instead of the 500 dpi size, so the DNN inputs have mismatched shapes and it raises. This
    does the resampling pyfing intends (same interpolations, periods scaled by 500/dpi), runs
    SNFEN at its native 500 dpi and resizes the result back, as pyfing does.
    """
    import cv2

    native = alg.parameters.dnn_input_dpi
    f = native / dpi
    h, w = image.shape
    img_n = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    hn, wn = img_n.shape
    mask_n = cv2.resize(mask, (wn, hn), interpolation=cv2.INTER_NEAREST)
    cos2 = cv2.resize(np.cos(orientation * 2), (wn, hn), interpolation=cv2.INTER_CUBIC)
    sin2 = cv2.resize(np.sin(orientation * 2), (wn, hn), interpolation=cv2.INTER_CUBIC)
    ori_n = np.arctan2(sin2, cos2) / 2
    per_n = cv2.resize(periods, (wn, hn), interpolation=cv2.INTER_CUBIC) * f
    en = alg.run(img_n, mask_n, ori_n, per_n, native)
    return cv2.resize(en, (w, h), interpolation=cv2.INTER_CUBIC)


def _check_gray8(arr: np.ndarray, what: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8 or arr.shape != shape:
        desc = f"{arr.dtype} {arr.shape}" if isinstance(arr, np.ndarray) else type(arr).__name__
        raise RuntimeError(f"pyfing {what} returned {desc}, expected uint8 {shape}")
    return np.ascontiguousarray(arr)


def preprocess(image: np.ndarray, attempt: AttemptConfig, *, configuration_id: str) -> PreprocessResult:
    """Run one attempt's pyfing chain on the raw sample (Req 5.1, 5.2, 5.8)."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a non-empty 2-D uint8 array")
    original_shape = (int(image.shape[0]), int(image.shape[1]))

    # resample (treated as native 500 ppi) -> optional CLAHE -> upscale (pyfing dpi = 500 * upscale)
    img = imaging.resize(image, attempt.resample)
    if attempt.clahe is not None:
        img = imaging.clahe(img, attempt.clahe.clip_limit, attempt.clahe.tile)
    img = imaging.resize(img, attempt.upscale)
    dpi = attempt.dpi
    shape = (int(img.shape[0]), int(img.shape[1]))

    timings: dict[str, float] = {}
    with _lock:  # cached algorithms carry mutable parameters (e.g. segmentation image_dpi)
        seg = _algorithm(attempt.segmentation)
        ori = _algorithm(attempt.orientation)
        frq = _algorithm(attempt.frequency)
        enh = _algorithm(attempt.enhancement)

        t = time.perf_counter_ns()
        seg.parameters.image_dpi = dpi
        mask = seg.run(img)
        timings["segmentation"] = (time.perf_counter_ns() - t) / 1e6
        mask = _check_gray8(mask, "segmentation", shape)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        t = time.perf_counter_ns()
        orientation = ori.run(img, mask, dpi)[0]
        timings["orientation"] = (time.perf_counter_ns() - t) / 1e6

        t = time.perf_counter_ns()
        periods = frq.run(img, mask, orientation, dpi)
        timings["frequency"] = (time.perf_counter_ns() - t) / 1e6

        t = time.perf_counter_ns()
        if attempt.enhancement == "SNFEN" and dpi != enh.parameters.dnn_input_dpi:
            enhanced = _snfen_resampled(enh, img, mask, orientation, periods, dpi)
        else:
            enhanced = enh.run(img, mask, orientation, periods, dpi)
        timings["enhancement"] = (time.perf_counter_ns() - t) / 1e6
        enhanced = _check_gray8(enhanced, "enhancement", shape)

    return PreprocessResult(
        attempt=attempt,
        enhanced=enhanced,
        mask=mask,
        scale_factor=attempt.scale_factor,
        original_shape=original_shape,
        configuration_id=configuration_id,
        timings_ms=timings,
    )


def quality_after_image(result: PreprocessResult) -> np.ndarray:
    """Inverted (dark ridges), masked (white background) enhanced image at the original 500 ppi size."""
    img = imaging.apply_mask(imaging.invert(result.enhanced), result.mask, background=255)
    return imaging.resize_to(img, result.original_shape)


def quality_after(result: PreprocessResult, settings: QualitySettings, *,
                  assess: Callable[..., QualityResult] = assess_quality) -> QualityResult:
    """NFIQ 2 score of the enhanced image on the same scale as quality_before (Req 5.5, 5.6)."""
    return assess(quality_after_image(result), settings)

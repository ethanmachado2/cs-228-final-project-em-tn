"""Image I/O and pixel operations shared by the quality and preprocessing stages.

All in-process images are 2-D `uint8` arrays. Across process or agent boundaries they are
serialized as 8-bit grayscale PNG bytes (`png_bytes` / `from_png_bytes`, Req 15.1).
"""

from __future__ import annotations

import struct
import tempfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PPI = 500
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_METERS_PER_INCH = 0.0254


class InputImageError(ValueError):
    """Raised when a sample cannot be read or decoded as an image (Req 2.5)."""


def _require_gray8(img: np.ndarray, name: str = "image") -> None:
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        desc = f"{img.dtype}, ndim={img.ndim}" if isinstance(img, np.ndarray) else type(img).__name__
        raise ValueError(f"{name} must be a 2-D uint8 array, got {desc}")
    if img.size == 0:
        raise ValueError(f"{name} is empty")


# --------------------------------------------------------------------------- decoding


def decode_gray8(data: bytes, *, source: str = "<bytes>") -> np.ndarray:
    """Decode encoded image bytes (BMP, PNG, ...) to a 2-D uint8 grayscale array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
    if img is None or img.ndim != 2 or img.size == 0:
        raise InputImageError(f"cannot decode {source} as an image")
    if img.dtype != np.uint8:  # IMREAD_GRAYSCALE yields 8-bit; guard against exotic decoders
        raise InputImageError(f"{source} decoded to {img.dtype}, expected uint8")
    return np.ascontiguousarray(img)


def load_gray8(path: Path | str) -> np.ndarray:
    """Load a sample as a 2-D uint8 grayscale array at its original size; raises InputImageError."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise InputImageError(f"cannot read {p}: {exc.strerror or exc}") from exc
    return decode_gray8(data, source=str(p))


# --------------------------------------------------------------------------- pixel operations


def clahe(img: np.ndarray, clip_limit: float, tile: int) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization with a `tile x tile` grid."""
    _require_gray8(img)
    if clip_limit <= 0 or tile < 1:
        raise ValueError(f"invalid CLAHE parameters clip_limit={clip_limit}, tile={tile}")
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile))).apply(img)


def scaled_size(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    """(H, W) after scaling by `scale`, rounded half-up and at least 1 pixel."""
    h, w = shape
    return max(1, int(np.floor(h * scale + 0.5))), max(1, int(np.floor(w * scale + 0.5)))


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Bicubic resize by `scale`; `scale == 1.0` returns an identical copy."""
    _require_gray8(img)
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    if scale == 1.0:
        return img.copy()
    h, w = scaled_size(img.shape, scale)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bicubic resize to an exact (H, W); same shape returns an identical copy."""
    _require_gray8(img)
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"target shape must be positive, got {shape!r}")
    if img.shape == (h, w):
        return img.copy()
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def invert(img: np.ndarray) -> np.ndarray:
    _require_gray8(img)
    return cv2.bitwise_not(img)


def apply_mask(img: np.ndarray, mask: np.ndarray, background: int = 255) -> np.ndarray:
    """Set pixels outside the foreground mask (mask == 0) to `background`."""
    _require_gray8(img)
    _require_gray8(mask, "mask")
    if mask.shape != img.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {img.shape}")
    if not 0 <= background <= 255:
        raise ValueError(f"background must be in 0..255, got {background}")
    out = img.copy()
    out[mask == 0] = background
    return out


# --------------------------------------------------------------------------- PNG encoding


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _with_phys(png: bytes, ppi: int) -> bytes:
    """Insert a pHYs chunk (pixels per metre) right after IHDR, replacing any existing one."""
    if not png.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG stream")
    ppm = int(round(ppi / _METERS_PER_INCH))
    phys = _chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))  # unit 1 = metre
    out = [_PNG_SIGNATURE]
    pos = len(_PNG_SIGNATURE)
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind = png[pos + 4 : pos + 8]
        end = pos + 12 + length
        if kind != b"pHYs":
            out.append(png[pos:end])
        if kind == b"IHDR":
            out.append(phys)
        pos = end
    return b"".join(out)


def png_bytes(img: np.ndarray, *, ppi: int = DEFAULT_PPI) -> bytes:
    """Encode as a lossless 8-bit grayscale PNG carrying `ppi` resolution metadata."""
    _require_gray8(img)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return _with_phys(buf.tobytes(), ppi)


def from_png_bytes(data: bytes) -> np.ndarray:
    """Decode PNG bytes produced by `png_bytes` back to a 2-D uint8 array."""
    if not data.startswith(_PNG_SIGNATURE):
        raise InputImageError("data is not a PNG stream")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or img.dtype != np.uint8:
        raise InputImageError("PNG is not an 8-bit grayscale image")
    return img


@contextmanager
def temp_png(img: np.ndarray, *, ppi: int = DEFAULT_PPI) -> Iterator[Path]:
    """Write `img` as a 500 PPI 8-bit grayscale PNG in a private temp dir, removed on exit."""
    data = png_bytes(img, ppi=ppi)
    with tempfile.TemporaryDirectory(prefix="fpv-") as tmp:  # mkdtemp => mode 0o700
        path = Path(tmp) / "image.png"
        path.write_bytes(data)
        yield path

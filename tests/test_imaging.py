"""Unit tests for fpv.imaging (Req 2.5, 4.6, 15.1)."""

from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

from fpv import imaging
from fpv.imaging import InputImageError


def _random_gray(seed: int, shape=(103, 96)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _phys(png: bytes) -> tuple[int, int, int] | None:
    pos = 8
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        if png[pos + 4 : pos + 8] == b"pHYs":
            return struct.unpack(">IIB", png[pos + 8 : pos + 8 + length])
        pos += 12 + length
    return None


@pytest.mark.parametrize("ext", [".BMP", ".bmp", ".png"])
def test_load_gray8_returns_2d_uint8_at_original_size(tmp_path, ext):
    img = _random_gray(0)
    path = tmp_path / f"sample{ext}"
    # A 3-channel file still loads as one 2-D gray channel.
    assert cv2.imwrite(str(path), cv2.merge([img, img, img]))
    out = imaging.load_gray8(path)
    assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == (103, 96)
    assert np.array_equal(out, img)


def test_load_gray8_raises_on_non_image_and_missing_file(tmp_path):
    bogus = tmp_path / "notes.BMP"
    bogus.write_text("not an image")
    with pytest.raises(InputImageError):
        imaging.load_gray8(bogus)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(InputImageError):
        imaging.load_gray8(empty)
    with pytest.raises(InputImageError):
        imaging.load_gray8(tmp_path / "missing.BMP")


@pytest.mark.parametrize("seed", range(5))
def test_png_round_trip_is_lossless(seed):
    img = _random_gray(seed, shape=(17 + seed * 13, 9 + seed * 7))
    data = imaging.png_bytes(img)
    back = imaging.from_png_bytes(data)
    assert back.dtype == np.uint8 and np.array_equal(back, img)


def test_png_carries_500_ppi_phys_chunk():
    data = imaging.png_bytes(_random_gray(1))
    assert _phys(data) == (19685, 19685, 1)  # 500 / 0.0254 pixels per metre
    # Exactly one pHYs chunk even if re-tagged.
    assert data.count(b"pHYs") == 1


def test_from_png_bytes_rejects_non_png():
    with pytest.raises(InputImageError):
        imaging.from_png_bytes(b"BM not a png")


def test_temp_png_writes_lossless_file_and_cleans_up():
    img = _random_gray(2)
    with imaging.temp_png(img) as path:
        assert path.suffix == ".png" and path.exists()
        assert (path.parent.stat().st_mode & 0o077) == 0  # private temp dir
        assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), img)
        assert _phys(path.read_bytes()) == (19685, 19685, 1)
    assert not path.exists() and not path.parent.exists()


def test_resize_identity_and_scaling():
    img = _random_gray(3)
    same = imaging.resize(img, 1.0)
    assert np.array_equal(same, img) and same is not img
    up = imaging.resize(img, 2.0)
    assert up.shape == (206, 192) and up.dtype == np.uint8
    assert imaging.resize(up, 0.5).shape == img.shape
    with pytest.raises(ValueError):
        imaging.resize(img, 0)


def test_invert_and_apply_mask():
    img = np.array([[0, 10], [200, 255]], np.uint8)
    assert np.array_equal(imaging.invert(img), 255 - img)
    mask = np.array([[255, 0], [0, 255]], np.uint8)
    assert np.array_equal(imaging.apply_mask(img, mask), np.array([[0, 255], [255, 255]], np.uint8))
    assert np.array_equal(imaging.apply_mask(img, mask, background=0), np.array([[0, 0], [0, 255]], np.uint8))
    with pytest.raises(ValueError):
        imaging.apply_mask(img, np.zeros((3, 3), np.uint8))


def test_clahe_preserves_shape_and_dtype():
    img = (np.arange(64 * 64).reshape(64, 64) % 40 + 100).astype(np.uint8)
    out = imaging.clahe(img, clip_limit=2.0, tile=8)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.std() > img.std()  # contrast was stretched


def test_non_gray8_inputs_are_rejected():
    with pytest.raises(ValueError):
        imaging.invert(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        imaging.png_bytes(np.zeros((4, 4), np.float32))

"""Unit tests for the `.xyt` template adapter (Req 7.1-7.4, 7.6). No toolchain needed."""

from __future__ import annotations

import math
import os
import random
from collections import namedtuple
from pathlib import Path

import pytest

from fpv import adapter as A
from fpv.adapter import XytMinutia, XytParseError, XytTemplate, read_xyt, to_xyt, write_xyt

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bozorth3"
SHAPE = (103, 96)  # SOCOFing (H, W)

# Same field layout as pyfing.Minutia(x, y, direction, type, quality).
M = namedtuple("M", "x y direction type quality", defaults=(0.0, "ending", 0.5))


def one(m, shape=SHAPE, s=1.0) -> XytMinutia:
    t = to_xyt([m], shape, s)
    assert t.count == 1
    return t.minutiae[0]


# --------------------------------------------------------------------------- conversion math


def test_y_axis_is_flipped_to_bottom_left_origin():
    assert (one(M(10, 0)).x, one(M(10, 0)).y) == (10, 102)
    assert (one(M(10, 102)).x, one(M(10, 102)).y) == (10, 0)


@pytest.mark.parametrize("direction, theta", [
    (0.0, 0), (math.pi / 2, 90), (-math.pi / 2, 270), (math.pi, 180), (-math.pi, 180),
    (math.radians(359.6), 0), (math.radians(-0.4), 0), (math.radians(45.2), 45),
])
def test_direction_to_theta(direction, theta):
    assert one(M(10, 10, direction)).theta == theta


@pytest.mark.parametrize("quality, q", [(0.0, 0), (1.0, 100), (1.2, 100), (-0.3, 0), (0.555, 56), (0.5, 50)])
def test_quality_scaled_and_clamped(quality, q):
    assert one(M(10, 10, quality=quality)).quality == q


def test_unscaling_at_s2():
    # (21, 41) in a 206-high enhanced image of a 103-high sample: x = 10.5 -> 11, y = 20.5 -> 21.
    m = one(M(21, 41), s=2.0)
    assert (m.x, m.y) == (11, 102 - 21)


def test_unscaling_at_s3():
    # (31, 62) in a 309-high enhanced image: x = 10.33 -> 10, y = 20.67 -> 21.
    m = one(M(31, 62), s=3.0)
    assert (m.x, m.y) == (10, 102 - 21)


def test_half_up_rounding_not_bankers():
    assert A.half_up(0.5) == 1 and A.half_up(2.5) == 3 and A.half_up(-0.5) == 0 and A.half_up(1.49) == 1
    assert one(M(1, 1), s=2.0).x == 1          # 0.5 -> 1 (Python's round would give 0)
    assert one(M(5, 0), s=2.0).x == 3          # 2.5 -> 3 (round would give 2)


@pytest.mark.parametrize("pt, s, expected", [
    ((-3, 10), 1.0, (0, 92)),          # left border
    ((1000, 10), 1.0, (95, 92)),       # right border: W - 1
    ((10, 400), 2.0, (5, 0)),          # below the bottom: y' clipped to 0
    ((10, -5), 2.0, (5, 102)),         # above the top: y' clipped to H - 1
    ((287, 308), 3.0, (95, 0)),        # last pixel of a 3x image -> bottom-right corner
])
def test_clipping_to_original_bounds(pt, s, expected):
    m = one(M(*pt), s=s)
    assert (m.x, m.y) == expected


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        to_xyt([M(1, 1)], SHAPE, 0)
    with pytest.raises(ValueError):
        to_xyt([M(1, 1)], (0, 96), 1.0)
    with pytest.raises(ValueError):
        to_xyt([M(float("nan"), 1)], SHAPE, 1.0)


# --------------------------------------------------------------------------- dedup + ordering


def test_dedup_keeps_highest_quality():
    t = to_xyt([M(10, 10, math.radians(10), quality=0.5), M(10, 10, math.radians(200), quality=0.8)], SHAPE, 1.0)
    assert t.minutiae == (XytMinutia(10, 92, 200, 80),)


def test_dedup_quality_tie_takes_lower_theta():
    t = to_xyt([M(20, 20, math.radians(90), quality=0.5), M(20, 20, math.radians(30), quality=0.5)], SHAPE, 1.0)
    assert t.minutiae == (XytMinutia(20, 82, 30, 50),)


def test_dedup_happens_after_unscaling():
    # Distinct enhanced-image pixels that land on one original pixel at s = 3.
    t = to_xyt([M(30, 30, quality=0.4), M(31, 29, quality=0.9), M(29, 31, quality=0.6)], SHAPE, 3.0)
    assert t.count == 1 and t.minutiae[0].quality == 90


def test_ordering_is_deterministic_by_y_x_theta():
    rng = random.Random(7)
    pts = [M(rng.uniform(0, 95), rng.uniform(0, 102), rng.uniform(-math.pi, math.pi), quality=rng.random())
           for _ in range(60)]
    ref = to_xyt(pts, SHAPE, 1.0)
    keys = [(m.y, m.x, m.theta) for m in ref.minutiae]
    assert keys == sorted(keys)
    assert len({(m.x, m.y) for m in ref.minutiae}) == ref.count
    for _ in range(5):
        rng.shuffle(pts)
        assert to_xyt(pts, SHAPE, 1.0) == ref


# --------------------------------------------------------------------------- rotation property


def _rotate_image_frame(x, y, direction, phi, cx, cy):
    """Rotate a minutia by phi counterclockwise *as seen on screen*, in image (y-down) coordinates."""
    dx, dy = x - cx, y - cy
    xr = cx + dx * math.cos(phi) + dy * math.sin(phi)
    yr = cy - dx * math.sin(phi) + dy * math.cos(phi)
    return xr, yr, math.atan2(math.sin(direction + phi), math.cos(direction + phi))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_shifts_theta_by_phi(seed):
    """Rotating the print by phi shifts every theta by phi mod 360 (+-1 deg) and rotates the
    adapted positions counterclockwise in the y-up `.xyt` frame. Guards the angle sign convention."""
    s, shape = 3.0, (400, 400)
    h, w = shape
    cx, cy = (w * s - 1) / 2, (h * s - 1) / 2         # center in enhanced-image pixels
    rng = random.Random(seed)
    base = []
    for _ in range(25):
        r, a = rng.uniform(0, 150 * s), rng.uniform(0, 2 * math.pi)
        base.append(M(cx + r * math.cos(a), cy + r * math.sin(a), rng.uniform(-math.pi, math.pi),
                      quality=rng.random()))
    for deg in range(0, 360, 15):
        phi = math.radians(deg)
        for m in base:
            ref = one(m, shape, s)
            xr, yr, dr = _rotate_image_frame(m.x, m.y, m.direction, phi, cx, cy)
            rot = one(M(xr, yr, dr, quality=m.quality), shape, s)
            shift = (rot.theta - ref.theta) % 360
            assert min((shift - deg) % 360, (deg - shift) % 360) <= 1, (deg, m, ref, rot)
            # Position: rotate ref about the center counterclockwise in the y-up frame.
            ox, oy = cx / s, (h - 1) - cy / s           # the same center in the .xyt frame
            ex = ox + (ref.x - ox) * math.cos(phi) - (ref.y - oy) * math.sin(phi)
            ey = oy + (ref.x - ox) * math.sin(phi) + (ref.y - oy) * math.cos(phi)
            assert abs(rot.x - ex) <= 1.5 and abs(rot.y - ey) <= 1.5, (deg, ref, rot, ex, ey)


# --------------------------------------------------------------------------- file I/O


def test_write_read_round_trip(tmp_path):
    rng = random.Random(3)
    pts = [M(rng.uniform(0, 287), rng.uniform(0, 308), rng.uniform(-math.pi, math.pi), quality=rng.random())
           for _ in range(40)]
    t = to_xyt(pts, SHAPE, 3.0)
    path = tmp_path / "probe.xyt"
    write_xyt(t, path)
    assert read_xyt(path) == t
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines == t.lines() and all(len(line.split()) == 4 for line in lines)
    assert [p.name for p in tmp_path.iterdir()] == ["probe.xyt"]      # no temp file left behind


def test_empty_template_round_trip(tmp_path):
    write_xyt(XytTemplate(()), tmp_path / "e.xyt")
    assert (tmp_path / "e.xyt").read_text() == "" and read_xyt(tmp_path / "e.xyt").count == 0


def test_write_is_atomic_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "t.xyt"
    path.write_text("1 2 3 4\n")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(A.os, "replace", boom)
    with pytest.raises(OSError):
        write_xyt(XytTemplate((XytMinutia(5, 6, 7, 8),)), path)
    assert path.read_text() == "1 2 3 4\n"
    assert sorted(os.listdir(tmp_path)) == ["t.xyt"]


def test_reads_bozorth3_fixture():
    t = read_xyt(FIXTURES / "handwritten_12.xyt")
    assert t.count == 12 and t.minutiae[0] == XytMinutia(10, 10, 0, 50)


@pytest.mark.parametrize("bad", ["1 2 3", "1 2 3 4 5", "1 2 x 4", "1 2 360 4", "1 2 3 101", "-1 2 3 4", "1.5 2 3 4"])
def test_corrupt_line_raises_parse_error(tmp_path, bad):
    path = tmp_path / "bad.xyt"
    path.write_text(f"10 10 0 50\n{bad}\n")
    with pytest.raises(XytParseError, match=r"bad\.xyt:2"):
        read_xyt(path)

"""Template adapter: pyfing minutiae -> NIST `.xyt` (Req 7).

pyfing gives `(x, y)` in pixels of the preprocessed (resampled/upscaled) image, top-left
origin, and `direction = atan2(-dy, dx)` in [-pi, pi], i.e. already counterclockwise with y up.
Bozorth3 (NIST internal convention, confirmed in the Task 1 spike) wants a bottom-left origin
and integer degrees, so with `s = scale_factor` and the original sample shape `(H, W)`:

    x'    = clip(round(x / s), 0, W - 1)
    y'    = clip((H - 1) - round(y / s), 0, H - 1)
    theta = round(degrees(direction)) mod 360        # round first: 359.6 -> 360 -> 0
    q     = clamp(round(quality * 100), 0, 100)

`round` is explicit half-up (`floor(v + 0.5)`), not Python's banker's rounding. The y flip
already converts to a y-up frame, so no angle negation is needed. Minutiae landing on the same
`(x', y')` are deduplicated (highest q, then lowest theta) and lines are sorted by
`(y', x', theta)` (Req 7.3, 7.4). This module has no pyfing/OpenCV dependency: any object with
`x`, `y`, `direction` and `quality` attributes is accepted.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ADAPTER_VERSION",
    "XytMinutia",
    "XytParseError",
    "XytTemplate",
    "half_up",
    "read_xyt",
    "to_xyt",
    "write_xyt",
]

ADAPTER_VERSION = "xyt-adapter/1"


class XytParseError(ValueError):
    """A `.xyt` file line is not four in-range integers `x y theta q`."""


def half_up(v: float) -> int:
    """Round half up (`floor(v + 0.5)`): 0.5 -> 1, 1.5 -> 2, -0.5 -> 0."""
    return math.floor(v + 0.5)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True, slots=True)
class XytMinutia:
    x: int
    y: int
    theta: int     # degrees, 0..359, counterclockwise, y up
    quality: int   # 0..100

    def __post_init__(self) -> None:
        for name in ("x", "y", "theta", "quality"):
            if not _is_int(getattr(self, name)):
                raise TypeError(f"XytMinutia.{name} must be an int, got {getattr(self, name)!r}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"XytMinutia coordinates must be non-negative, got ({self.x}, {self.y})")
        if not 0 <= self.theta < 360:
            raise ValueError(f"XytMinutia.theta must be in 0..359, got {self.theta}")
        if not 0 <= self.quality <= 100:
            raise ValueError(f"XytMinutia.quality must be in 0..100, got {self.quality}")

    def line(self) -> str:
        return f"{self.x} {self.y} {self.theta} {self.quality}"


@dataclass(frozen=True, slots=True)
class XytTemplate:
    minutiae: tuple[XytMinutia, ...]
    adapter_version: str = ADAPTER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "minutiae", tuple(self.minutiae))
        for m in self.minutiae:
            if not isinstance(m, XytMinutia):
                raise TypeError(f"XytTemplate.minutiae must hold XytMinutia, got {type(m).__name__}")

    @property
    def count(self) -> int:
        return len(self.minutiae)

    def lines(self) -> list[str]:
        """One `x y theta q` line per minutia, in template order."""
        return [m.line() for m in self.minutiae]

    def to_text(self) -> str:
        lines = self.lines()
        return "\n".join(lines) + "\n" if lines else ""


# --------------------------------------------------------------------------- conversion


def _finite(name: str, v: Any) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"minutia {name} must be finite, got {v!r}")
    return f


def convert(m: Any, original_shape: tuple[int, int], scale_factor: float) -> XytMinutia:
    """Convert one pyfing minutia (Req 7.1, 7.2); no dedup."""
    h, w = original_shape
    x = _finite("x", m.x) / scale_factor
    y = _finite("y", m.y) / scale_factor
    theta = half_up(math.degrees(_finite("direction", m.direction))) % 360
    q = half_up(_finite("quality", m.quality) * 100)
    return XytMinutia(
        x=min(max(half_up(x), 0), w - 1),
        y=min(max((h - 1) - half_up(y), 0), h - 1),
        theta=int(theta),
        quality=min(max(q, 0), 100),
    )


def dedup_and_sort(minutiae: Iterable[XytMinutia]) -> tuple[XytMinutia, ...]:
    """One minutia per `(x, y)`: highest quality, then lowest theta; sorted by `(y, x, theta)`."""
    best: dict[tuple[int, int], XytMinutia] = {}
    for m in minutiae:
        cur = best.get((m.x, m.y))
        if cur is None or (m.quality, -m.theta) > (cur.quality, -cur.theta):
            best[(m.x, m.y)] = m
    return tuple(sorted(best.values(), key=lambda m: (m.y, m.x, m.theta)))


def to_xyt(minutiae: Sequence[Any], original_shape: tuple[int, int], scale_factor: float) -> XytTemplate:
    """Adapt pyfing minutiae (enhanced-image pixels) to a `.xyt` template of the original sample."""
    if len(original_shape) != 2 or not all(_is_int(v) and v > 0 for v in original_shape):
        raise ValueError(f"original_shape must be (H, W) positive ints, got {original_shape!r}")
    s = float(scale_factor)
    if not math.isfinite(s) or s <= 0:
        raise ValueError(f"scale_factor must be a positive number, got {scale_factor!r}")
    return XytTemplate(dedup_and_sort(convert(m, original_shape, s) for m in minutiae))


# --------------------------------------------------------------------------- file I/O


def write_xyt(t: XytTemplate, path: Path) -> None:
    """Write `t` atomically: a temp file in the target directory, fsync, then `os.replace`."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as fh:
            fh.write(t.to_text())
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def parse_xyt(text: str, *, source: str = "<string>") -> XytTemplate:
    """Parse `.xyt` text (four integers per line; blank lines ignored), keeping line order."""
    out: list[XytMinutia] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 4:
            raise XytParseError(f"{source}:{n}: expected 4 integers 'x y theta q', got {raw!r}")
        try:
            out.append(XytMinutia(*(int(p, 10) for p in parts)))
        except (ValueError, TypeError) as exc:
            raise XytParseError(f"{source}:{n}: invalid minutia {raw!r}: {exc}") from exc
    return XytTemplate(tuple(out))


def read_xyt(path: Path) -> XytTemplate:
    """Read a `.xyt` file written by `write_xyt` (or any 4-column NIST `.xyt`)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise XytParseError(f"{path}: not an ASCII .xyt file") from exc
    return parse_xyt(text, source=str(path))

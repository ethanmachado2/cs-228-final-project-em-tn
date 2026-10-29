"""Bozorth3 matcher (Req 9.3-9.7).

`match` reads both `.xyt` templates, refuses to call Bozorth3 when either side has fewer than
`MIN_BOZORTH_MINUTIAE` minutiae (Bozorth3 itself silently returns 0 for such templates, see the
Task 1 spike), then runs `[bozorth3_bin, probe, gallery]` as an argument list (never a shell)
with a timeout and parses the single integer score Bozorth3 prints on stdout. The default NIST
internal `.xyt` convention is used (no `-m1`).

Template read errors (`OSError`, `adapter.XytParseError`) propagate unchanged so callers can
tell bad input apart from a tool failure; every Bozorth3 failure raises `MatcherError`.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fpv.adapter import read_xyt
from fpv.types import Status

__all__ = [
    "MIN_BOZORTH_MINUTIAE",
    "InsufficientMinutiae",
    "MatchResult",
    "MatcherError",
    "bozorth3_command",
    "decide",
    "match",
    "parse_bozorth3_output",
]

MIN_BOZORTH_MINUTIAE = 10

Runner = Callable[..., subprocess.CompletedProcess]
_SCORE_RE = re.compile(r"^\d+$")


@dataclass(frozen=True, slots=True)
class MatchResult:
    score: int
    probe_count: int
    gallery_count: int
    tool_version: str
    duration_ms: float


class InsufficientMinutiae(Exception):
    """A template has fewer than `minimum` minutiae; Bozorth3 was not invoked (Req 9.5)."""

    def __init__(self, which: Literal["probe", "gallery"], count: int,
                 minimum: int = MIN_BOZORTH_MINUTIAE) -> None:
        super().__init__(f"{which} template has {count} minutiae; Bozorth3 needs at least {minimum}")
        self.which = which
        self.count = count
        self.minimum = minimum


class MatcherError(Exception):
    """Bozorth3 is missing, timed out, exited non-zero or printed unparseable output (Req 9.6)."""


def bozorth3_command(bozorth3_bin: str, probe: Path, gallery: Path) -> list[str]:
    return [bozorth3_bin, str(probe), str(gallery)]


def parse_bozorth3_output(stdout: str) -> int:
    """Bozorth3 prints exactly one non-negative integer for a single probe/gallery pair."""
    text = (stdout or "").strip()
    if not _SCORE_RE.match(text):
        raise MatcherError(f"unparseable Bozorth3 output: {text[:200]!r}")
    return int(text)


def match(probe: Path, gallery: Path, *, bozorth3_bin: str = "bozorth3", timeout_s: float = 30,
          tool_version: str = "unknown", runner: Runner = subprocess.run) -> MatchResult:
    """Bozorth3 score of `probe` against `gallery` (both `.xyt` files)."""
    probe, gallery = Path(probe), Path(gallery)
    probe_count = read_xyt(probe).count
    gallery_count = read_xyt(gallery).count
    if probe_count < MIN_BOZORTH_MINUTIAE:
        raise InsufficientMinutiae("probe", probe_count)
    if gallery_count < MIN_BOZORTH_MINUTIAE:
        raise InsufficientMinutiae("gallery", gallery_count)

    cmd = bozorth3_command(bozorth3_bin, probe, gallery)
    start = time.perf_counter_ns()
    try:
        proc = runner(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                      timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise MatcherError(f"Bozorth3 binary not found: {bozorth3_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MatcherError(f"Bozorth3 timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise MatcherError(f"cannot run Bozorth3: {exc}") from exc
    duration_ms = (time.perf_counter_ns() - start) / 1e6

    if proc.returncode != 0:
        how = f"signal {-proc.returncode}" if proc.returncode < 0 else f"exit code {proc.returncode}"
        detail = (proc.stderr or proc.stdout or "").strip()[:200]
        raise MatcherError(f"Bozorth3 terminated abnormally ({how}): {detail}")
    return MatchResult(
        score=parse_bozorth3_output(proc.stdout),
        probe_count=probe_count,
        gallery_count=gallery_count,
        tool_version=tool_version,
        duration_ms=duration_ms,
    )


def decide(score: int, threshold: int) -> Literal[Status.MATCH, Status.NO_MATCH]:
    """`MATCH` iff `score >= threshold` (Req 9.4)."""
    return Status.MATCH if score >= threshold else Status.NO_MATCH

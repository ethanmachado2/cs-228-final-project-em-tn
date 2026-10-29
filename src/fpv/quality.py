"""NFIQ 2 quality assessment and the raw quality gate (Req 4).

The NFIQ 2 CLI is run as `[nfiq2, "-F", "-a", <png>]` with `shell=False` and `stdin=DEVNULL`
(Task 1 spike): `-F` stops the interactive PPI prompt, `-a` emits a CSV row with the
actionable feedback. NFIQ 2 exits 0 even when it cannot score an image; that case shows up as
`QualityScore=NA` and maps to `score=None`. Only a missing binary, a timeout, a non-zero exit
(including a signal) or unparseable output raises `QualityToolError`.

NFIQ 2 cannot score native 96x103 SOCOFing images, so every image is bicubic-upscaled by the
configured `quality.nfiq2_upscale` (default 3.0) before scoring. The same settings are reused
for `quality_after` (Req 5.5), so both scores live on the same scale.
"""

from __future__ import annotations

import csv
import io
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fpv import imaging

if TYPE_CHECKING:
    from fpv.config import PipelineConfig

NFIQ2_FLAGS = ("-F", "-a")
ACTIONABLE_FIELDS = (
    "UniformImage",
    "EmptyImageOrContrastTooLow",
    "FingerprintImageWithMinutiae",
    "SufficientFingerprintForeground",
)
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_NFIQ2_FAILED = "nfiq2_failed"

Runner = Callable[..., subprocess.CompletedProcess]


class QualityToolError(Exception):
    """NFIQ 2 is missing, timed out, crashed or produced unparseable output (-> Status.ERROR)."""


@dataclass(frozen=True, slots=True)
class QualitySettings:
    """How NFIQ 2 is invoked. Build it from the config so quality_before/after share one scale."""

    upscale: float
    nfiq2_bin: str = "nfiq2"
    timeout_s: float = 30.0
    tool_version: str = "unknown"

    def __post_init__(self) -> None:
        if not self.upscale > 0:
            raise ValueError(f"upscale must be positive, got {self.upscale!r}")
        if not self.timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s!r}")

    @classmethod
    def from_config(cls, conf: PipelineConfig, **overrides: object) -> QualitySettings:
        values: dict[str, object] = {
            "upscale": conf.nfiq2_upscale,
            "tool_version": conf.tools.get("nfiq2", "unknown"),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class QualityResult:
    score: int | None                       # None => NFIQ 2 could not score (Req 4.3 / 5.6)
    actionable_feedback: Mapping[str, float]
    raw_output: str
    tool_version: str
    duration_ms: float
    upscale: float = 1.0                    # factor applied before scoring
    error: str | None = None                # NFIQ 2 OptionalError text when score is None
    scored_shape: tuple[int, int] = (0, 0)  # (H, W) of the PNG given to NFIQ 2


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    score: int | None
    threshold: int
    reason: str | None                      # "below_threshold" | "nfiq2_failed" | None


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    score: int | None
    actionable_feedback: Mapping[str, float]
    error: str | None


def _na(value: str | None) -> bool:
    return value is None or value.strip() in ("", "NA")


def parse_nfiq2_output(stdout: str) -> ParsedOutput:
    """Parse `nfiq2 -a` CSV output (header + one row) into score, feedback and error.

    Raises QualityToolError if the output is not the expected CSV.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith('"Filename"')), None)
    if header_idx is None or header_idx + 1 >= len(lines):
        raise QualityToolError(f"unexpected NFIQ 2 output (no CSV header/row): {stdout[:200]!r}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header_idx:header_idx + 2]))))
    if len(rows) != 1 or "QualityScore" not in rows[0]:
        raise QualityToolError(f"unexpected NFIQ 2 CSV: {stdout[:200]!r}")
    row = rows[0]

    raw_score = row["QualityScore"]
    error = None if _na(row.get("OptionalError")) else row["OptionalError"].strip()
    if _na(raw_score):
        return ParsedOutput(score=None, actionable_feedback={}, error=error or "QualityScore=NA")
    try:
        score = int(raw_score)
    except ValueError as exc:
        raise QualityToolError(f"unparseable NFIQ 2 QualityScore {raw_score!r}") from exc
    if not 0 <= score <= 100:
        raise QualityToolError(f"NFIQ 2 QualityScore {score} outside 0..100")

    feedback: dict[str, float] = {}
    for name in ACTIONABLE_FIELDS:
        value = row.get(name)
        if _na(value):
            continue
        try:
            feedback[name] = float(value)
        except ValueError as exc:
            raise QualityToolError(f"unparseable NFIQ 2 feedback {name}={value!r}") from exc
    return ParsedOutput(score=score, actionable_feedback=feedback, error=error)


def nfiq2_command(settings: QualitySettings, png: str) -> list[str]:
    return [settings.nfiq2_bin, *NFIQ2_FLAGS, png]


def assess_quality(image: np.ndarray, settings: QualitySettings, *,
                   runner: Runner = subprocess.run) -> QualityResult:
    """Score `image` with NFIQ 2 after a bicubic upscale by `settings.upscale` (Req 4.1, 4.5, 4.6)."""
    scaled = imaging.resize(image, settings.upscale)
    start = time.perf_counter_ns()
    with imaging.temp_png(scaled) as png:
        cmd = nfiq2_command(settings, str(png))
        try:
            proc = runner(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=settings.timeout_s, check=False)
        except FileNotFoundError as exc:
            raise QualityToolError(f"NFIQ 2 binary not found: {settings.nfiq2_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise QualityToolError(f"NFIQ 2 timed out after {settings.timeout_s}s") from exc
        except OSError as exc:
            raise QualityToolError(f"cannot run NFIQ 2: {exc}") from exc
    duration_ms = (time.perf_counter_ns() - start) / 1e6

    stdout = proc.stdout or ""
    if proc.returncode != 0:
        how = f"signal {-proc.returncode}" if proc.returncode < 0 else f"exit code {proc.returncode}"
        detail = (proc.stderr or stdout).strip()[:200]
        raise QualityToolError(f"NFIQ 2 terminated abnormally ({how}): {detail}")
    parsed = parse_nfiq2_output(stdout)
    return QualityResult(
        score=parsed.score,
        actionable_feedback=parsed.actionable_feedback,
        raw_output=stdout,
        tool_version=settings.tool_version,
        duration_ms=duration_ms,
        upscale=settings.upscale,
        error=parsed.error,
        scored_shape=(int(scaled.shape[0]), int(scaled.shape[1])),
    )


def quality_gate(result: QualityResult, threshold: int) -> GateDecision:
    """Pass iff NFIQ 2 produced a score >= threshold (Req 4.2, 4.3)."""
    if result.score is None:
        return GateDecision(False, None, threshold, REASON_NFIQ2_FAILED)
    if result.score < threshold:
        return GateDecision(False, result.score, threshold, REASON_BELOW_THRESHOLD)
    return GateDecision(True, result.score, threshold, None)

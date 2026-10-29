"""Versioned pipeline configuration: YAML loading, validation and identifiers.

Identifiers (design "Identifier scheme"):
  canonical_json    json.dumps(sort_keys, compact separators, ascii, no NaN)
  configuration_id  "cfg_" + sha256(canonical_json({attempt, adapter, tools}))[:12]
  config_set_id     "cs_"  + sha256(canonical_json(config minus provenance))[:12]
  pipeline_version  "<version>+g<short sha>[.dirty]" or "<version>+unknown"
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml

from fpv import __version__

DEFAULT_CONFIG = Path("configs/pipeline.v1.yaml")
FROZEN_CONFIG = Path("configs/pipeline.v1.frozen.yaml")
SCHEMA_VERSION = 1
HASH_LEN = 12
# Keys excluded from config_set_id (metadata about the config, not parameters of it).
NON_HASHED_KEYS = frozenset({"provenance"})
REQUIRED_TOOL_PINS = ("nfiq2", "nbis", "pyfing")
REQUIRED_PATHS = ("raw", "splits", "exclusions", "templates", "probes", "db", "reports")
# Allowed range of the fixed bicubic upscale applied before NFIQ 2 (quality.nfiq2_upscale).
NFIQ2_UPSCALE_RANGE = (1.0, 4.0)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or out of range."""


# --------------------------------------------------------------------------- canonical JSON / ids


def _normalize(obj: Any, where: str = "$") -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ConfigError(f"{where}: non-finite float {obj!r} cannot be hashed")
        return float(obj)
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ConfigError(f"{where}: mapping keys must be strings, got {k!r}")
            out[k] = _normalize(v, f"{where}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    raise ConfigError(f"{where}: unsupported value type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, ASCII, floats via repr."""
    return json.dumps(
        _normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(prefix: str, obj: Any) -> str:
    return prefix + hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()[:HASH_LEN]


def configuration_id(
    attempt: AttemptConfig | Mapping[str, Any], adapter_version: str, tools: Mapping[str, Any]
) -> str:
    """Per-attempt id over the attempt configuration, adapter version and pinned tools (Req 12.3).

    An `AttemptConfig` is hashed through `to_dict()`, i.e. with defaults filled in and numbers
    normalized, so equivalent YAML spellings (`upscale: 2` vs `2.0`, omitted defaults) agree.
    """
    if isinstance(attempt, AttemptConfig):
        attempt = attempt.to_dict()
    return _digest("cfg_", {"attempt": attempt, "adapter": adapter_version, "tools": tools})


def config_set_id(raw: Mapping[str, Any]) -> str:
    """Id over the entire configuration (including thresholds), minus provenance (Req 12.4)."""
    return _digest("cs_", {k: v for k, v in raw.items() if k not in NON_HASHED_KEYS})


# --------------------------------------------------------------------------- pipeline_version

_SHA_RE = re.compile(r"^([0-9a-fA-F]{7,40})(\.dirty)?$")
GitProbe = Callable[[], "tuple[str, bool] | None"]


def _git_probe(cwd: Path | None = None) -> tuple[str, bool] | None:
    """Return (short sha, dirty) from the git work tree, or None if unavailable."""
    cwd = cwd or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not sha:
        return None
    return sha, bool(status.strip())


def pipeline_version(
    *,
    env: Mapping[str, str] | None = None,
    git: GitProbe | None = None,
    version: str = __version__,
) -> str:
    """`<version>+g<sha>[.dirty]`; GIT_COMMIT (container build arg) wins over the git work tree."""
    env = os.environ if env is None else env
    commit = (env.get("GIT_COMMIT") or "").strip()
    if commit and commit.lower() != "unknown":
        m = _SHA_RE.match(commit)
        if m:
            return f"{version}+g{m.group(1).lower()}{'.dirty' if m.group(2) else ''}"
    probe = (git or _git_probe)()
    if probe is not None:
        sha, dirty = probe
        if _SHA_RE.match(sha):
            return f"{version}+g{sha.lower()}{'.dirty' if dirty else ''}"
    return f"{version}+unknown"


# --------------------------------------------------------------------------- attempt configs
# Defined here (not in preprocess.py) so that loading a config never imports OpenCV/pyfing;
# preprocess.py re-exports them as part of its public contract.

SEGMENTATION_METHODS = ("GMFS", "SUFS")
ORIENTATION_METHODS = ("GBFOE", "SNFOE")
FREQUENCY_METHODS = ("XSFFE", "SNFFE")
ENHANCEMENT_METHODS = ("GBFEN", "SNFEN")
# pyfing 0.7.3 methods that only work at 500 dpi (Xsffe raises; Gbfen does not rescale its
# orientation/period maps), so they cannot be combined with an upscale.
NATIVE_DPI_ONLY = frozenset({"XSFFE", "GBFEN"})
ATTEMPT_UPSCALE_RANGE = (1.0, 4.0)
ATTEMPT_RESAMPLE_RANGE = (1.0, 4.0)
_ATTEMPT_KEYS = frozenset({"attempt_no", "name", "segmentation", "orientation", "frequency",
                           "enhancement", "clahe", "resample", "upscale", "sbmex_overrides"})


@dataclass(frozen=True, slots=True)
class ClaheConfig:
    clip_limit: float = 2.0
    tile: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.clip_limit, bool) or not isinstance(self.clip_limit, (int, float)) \
                or not math.isfinite(self.clip_limit) or self.clip_limit <= 0:
            raise ConfigError(f"clahe.clip_limit must be a positive number, got {self.clip_limit!r}")
        if isinstance(self.tile, bool) or not isinstance(self.tile, int) or self.tile < 1:
            raise ConfigError(f"clahe.tile must be a positive integer, got {self.tile!r}")
        object.__setattr__(self, "clip_limit", float(self.clip_limit))

    def to_dict(self) -> dict[str, Any]:
        return {"clip_limit": self.clip_limit, "tile": self.tile}


def _frozen_overrides(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(sorted((value or {}).items())))


@dataclass(frozen=True, slots=True)
class AttemptConfig:
    """One deterministic preprocessing attempt (Req 5.2). Hashed into its configuration_id."""

    attempt_no: int
    name: str                                         # "traditional" | "deep" | "deep_boost"
    segmentation: Literal["GMFS", "SUFS"]
    orientation: Literal["GBFOE", "SNFOE"]
    frequency: Literal["XSFFE", "SNFFE"]
    enhancement: Literal["GBFEN", "SNFEN"]
    clahe: ClaheConfig | None = None
    upscale: float = 1.0                              # passes dpi = 500 * upscale to pyfing
    sbmex_overrides: Mapping[str, float | int | str] = field(default_factory=dict)
    # Bicubic resample applied first, with the result treated as a native 500 ppi image (pyfing
    # is not told about it). SOCOFing ridges are ~3-6 px apart, so pyfing needs this to see a
    # 500 ppi-like ridge period (Task 5 characterization). Total scale = resample * upscale.
    resample: float = 1.0

    def __post_init__(self) -> None:
        where = f"attempt {self.attempt_no!r}"
        if isinstance(self.attempt_no, bool) or not isinstance(self.attempt_no, int) or self.attempt_no < 1:
            raise ConfigError(f"attempt_no must be a positive integer, got {self.attempt_no!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError(f"{where}: name must be a non-empty string")
        for stage, allowed in (("segmentation", SEGMENTATION_METHODS), ("orientation", ORIENTATION_METHODS),
                               ("frequency", FREQUENCY_METHODS), ("enhancement", ENHANCEMENT_METHODS)):
            value = getattr(self, stage)
            if value not in allowed:
                raise ConfigError(f"{where}: unknown {stage} method {value!r} (allowed: {', '.join(allowed)})")
        if self.clahe is not None and not isinstance(self.clahe, ClaheConfig):
            raise ConfigError(f"{where}: clahe must be a ClaheConfig or null")
        lo, hi = ATTEMPT_UPSCALE_RANGE
        if isinstance(self.upscale, bool) or not isinstance(self.upscale, (int, float)) \
                or not lo <= self.upscale <= hi:
            raise ConfigError(f"{where}: upscale must be a number in {lo}..{hi}, got {self.upscale!r}")
        object.__setattr__(self, "upscale", float(self.upscale))
        lo, hi = ATTEMPT_RESAMPLE_RANGE
        if isinstance(self.resample, bool) or not isinstance(self.resample, (int, float)) \
                or not lo <= self.resample <= hi:
            raise ConfigError(f"{where}: resample must be a number in {lo}..{hi}, got {self.resample!r}")
        object.__setattr__(self, "resample", float(self.resample))
        if self.upscale != 1.0:
            native = [m for m in (self.frequency, self.enhancement) if m in NATIVE_DPI_ONLY]
            if native:
                raise ConfigError(f"{where}: {', '.join(native)} only support 500 dpi; upscale must be 1.0")
        if not isinstance(self.sbmex_overrides, Mapping):
            raise ConfigError(f"{where}: sbmex_overrides must be a mapping")
        for k, v in self.sbmex_overrides.items():
            if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float, str)):
                raise ConfigError(f"{where}: sbmex_overrides.{k} must be a number or string, got {v!r}")
        object.__setattr__(self, "sbmex_overrides", _frozen_overrides(self.sbmex_overrides))

    @property
    def dpi(self) -> int | float:
        """Resolution passed to pyfing for the upscaled image."""
        dpi = 500 * self.upscale
        return int(dpi) if float(dpi).is_integer() else dpi

    @property
    def scale_factor(self) -> float:
        """Size of the preprocessed images relative to the raw sample (undone by the adapter)."""
        return self.resample * self.upscale

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, fully explicit form (defaults included); this is what configuration_id hashes."""
        return {
            "attempt_no": self.attempt_no,
            "name": self.name,
            "segmentation": self.segmentation,
            "orientation": self.orientation,
            "frequency": self.frequency,
            "enhancement": self.enhancement,
            "clahe": self.clahe.to_dict() if self.clahe is not None else None,
            "resample": self.resample,
            "upscale": self.upscale,
            "sbmex_overrides": dict(self.sbmex_overrides),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AttemptConfig:
        if not isinstance(d, Mapping):
            raise ConfigError(f"attempt must be a mapping, got {type(d).__name__}")
        unknown = sorted(set(d) - _ATTEMPT_KEYS)
        if unknown:
            raise ConfigError(f"attempt {d.get('attempt_no')!r}: unknown keys {unknown}")
        missing = [k for k in ("attempt_no", "name", "segmentation", "orientation", "frequency", "enhancement")
                   if k not in d]
        if missing:
            raise ConfigError(f"attempt {d.get('attempt_no')!r}: missing keys {missing}")
        clahe = d.get("clahe")
        if clahe is not None:
            if not isinstance(clahe, Mapping) or set(clahe) - {"clip_limit", "tile"}:
                raise ConfigError(f"attempt {d['attempt_no']!r}: clahe must be a mapping of clip_limit/tile")
            clahe = ClaheConfig(**clahe)
        return cls(
            attempt_no=d["attempt_no"], name=d["name"], segmentation=d["segmentation"],
            orientation=d["orientation"], frequency=d["frequency"], enhancement=d["enhancement"],
            clahe=clahe, upscale=d.get("upscale", 1.0), sbmex_overrides=d.get("sbmex_overrides") or {},
            resample=d.get("resample", 1.0),
        )


def parse_attempts(items: Any, max_attempts: int) -> tuple[AttemptConfig, ...]:
    """Parse `preprocess.attempts`: at most 3 entries numbered exactly 1..n, n >= max_attempts."""
    if not isinstance(items, list) or not items:
        raise ConfigError("preprocess.attempts must be a non-empty list of attempt mappings")
    if len(items) > 3:
        raise ConfigError(f"preprocess.attempts has {len(items)} entries (at most 3 allowed)")
    try:
        attempts = tuple(AttemptConfig.from_dict(a) for a in items)
    except ConfigError as exc:
        raise ConfigError(f"preprocess.attempts: {exc}") from exc
    numbers = [a.attempt_no for a in attempts]
    if numbers != list(range(1, len(attempts) + 1)):
        raise ConfigError(f"preprocess.attempts attempt_no must be 1..{len(attempts)} in order, got {numbers}")
    if max_attempts > len(attempts):
        raise ConfigError(f"preprocess.max_attempts={max_attempts} exceeds the {len(attempts)} defined attempts")
    return attempts


# --------------------------------------------------------------------------- typed config


@dataclass(frozen=True, slots=True)
class Thresholds:
    nfiq2_min: int
    minutiae_min: int
    bozorth3_match: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    raw: Mapping[str, Any]
    source_path: Path
    root: Path
    thresholds: Thresholds
    max_presentations: int
    nfiq2_upscale: float
    max_attempts: int
    attempts: tuple[AttemptConfig, ...]
    adapter_version: str
    tools: Mapping[str, str]
    paths: Mapping[str, str]
    frozen: bool
    config_set_id: str = field(default="")

    def attempt(self, attempt_no: int) -> AttemptConfig:
        """The attempt numbered `attempt_no` (1-based); KeyError if it is not defined."""
        if isinstance(attempt_no, bool) or not 1 <= attempt_no <= len(self.attempts):
            raise KeyError(f"attempt {attempt_no!r} is not defined (1..{len(self.attempts)})")
        return self.attempts[attempt_no - 1]

    def configuration_id(self, attempt: AttemptConfig | int) -> str:
        """configuration_id of an attempt under this config's adapter version and tool pins."""
        if not isinstance(attempt, AttemptConfig):
            attempt = self.attempt(attempt)
        return configuration_id(attempt, self.adapter_version, self.tools)

    def path(self, key: str) -> Path:
        """Resolve a configured data path relative to the project root."""
        p = Path(self.paths[key])
        return p if p.is_absolute() else self.root / p


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing or invalid section '{key}' (expected a mapping)")
    return value


def _int(section: Mapping[str, Any], key: str, where: str, *, lo: int | None = None, hi: int | None = None) -> int:
    if key not in section:
        raise ConfigError(f"{where}.{key} is required")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        rng = f"{lo if lo is not None else '-inf'}..{hi if hi is not None else 'inf'}"
        raise ConfigError(f"{where}.{key}={value} is out of range (allowed {rng})")
    return value


def validate(raw: Any, *, source_path: Path = Path("<memory>"), root: Path | None = None) -> PipelineConfig:
    """Validate a parsed YAML document and build a PipelineConfig (Req 12.1, 12.2)."""
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration must be a YAML mapping")
    raw = copy.deepcopy(dict(raw))
    canonical_json(raw)  # rejects unhashable values early

    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}, got {schema!r}")
    frozen = raw.get("frozen", False)
    if not isinstance(frozen, bool):
        raise ConfigError(f"frozen must be a boolean, got {frozen!r}")

    th = _section(raw, "thresholds")
    thresholds = Thresholds(
        nfiq2_min=_int(th, "nfiq2_min", "thresholds", lo=0, hi=100),
        minutiae_min=_int(th, "minutiae_min", "thresholds", lo=10),
        bozorth3_match=_int(th, "bozorth3_match", "thresholds", lo=0),
    )
    max_presentations = _int(_section(raw, "session"), "max_presentations", "session", lo=1)

    quality = _section(raw, "quality")
    upscale = quality.get("nfiq2_upscale")
    lo, hi = NFIQ2_UPSCALE_RANGE
    if isinstance(upscale, bool) or not isinstance(upscale, (int, float)):
        raise ConfigError(f"quality.nfiq2_upscale must be a number, got {upscale!r}")
    if not lo <= upscale <= hi:
        raise ConfigError(f"quality.nfiq2_upscale={upscale} is out of range (allowed {lo}..{hi})")

    pre = _section(raw, "preprocess")
    max_attempts = _int(pre, "max_attempts", "preprocess", lo=1, hi=3)
    attempts = parse_attempts(pre.get("attempts"), max_attempts)

    adapter_version = _section(raw, "adapter").get("version")
    if not isinstance(adapter_version, str) or not adapter_version:
        raise ConfigError("adapter.version must be a non-empty string")

    tools = _section(raw, "tools")
    for name in REQUIRED_TOOL_PINS:
        if not isinstance(tools.get(name), str) or not tools.get(name):
            raise ConfigError(f"tools.{name} must be a pinned version string")
    for name, value in tools.items():
        if not isinstance(value, str):
            raise ConfigError(f"tools.{name} must be a string (quote versions in YAML), got {value!r}")

    paths = _section(raw, "paths")
    for name in REQUIRED_PATHS:
        if not isinstance(paths.get(name), str) or not paths.get(name):
            raise ConfigError(f"paths.{name} must be a non-empty string")

    return PipelineConfig(
        raw=raw,
        source_path=source_path,
        root=(root or Path.cwd()).resolve(),
        thresholds=thresholds,
        max_presentations=max_presentations,
        nfiq2_upscale=float(upscale),
        max_attempts=max_attempts,
        attempts=attempts,
        adapter_version=adapter_version,
        tools=dict(tools),
        paths=dict(paths),
        frozen=frozen,
        config_set_id=config_set_id(raw),
    )


def resolve_config_path(explicit: str | Path | None, root: Path | None = None) -> Path:
    """`--config` if given, else the frozen config if present, else the default (Req 12.1, 12.7)."""
    if explicit is not None:
        return Path(explicit)
    root = root or Path.cwd()
    frozen = root / FROZEN_CONFIG
    return frozen if frozen.exists() else root / DEFAULT_CONFIG


def load(path: str | Path | None = None, *, root: Path | None = None) -> PipelineConfig:
    """Load and validate a configuration file; raises ConfigError with a descriptive message."""
    root = root or Path.cwd()
    cfg_path = resolve_config_path(path, root)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {cfg_path}: {exc.strerror or exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {cfg_path}: {exc}") from exc
    try:
        return validate(raw, source_path=cfg_path, root=root)
    except ConfigError as exc:
        raise ConfigError(f"{cfg_path}: {exc}") from exc

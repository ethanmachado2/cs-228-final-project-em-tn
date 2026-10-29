"""SQLite metadata store plus the `.xyt` artifact store (Req 8, 9.1, 9.8, 10.5, 11).

`Repository` owns one SQLite connection (WAL, `foreign_keys=ON`, autocommit mode with explicit
`BEGIN IMMEDIATE` transactions) and two artifact roots:

    <templates_root>/<user_id>/<Hand>_<finger>/v<n>.xyt     enrolled templates (Req 8.1)
    <probes_root>/<run_id>.xyt                               verification probes (Req 9.1)

Artifact locations are stored relative to `artifact_root` (the project root when built with
`from_config`) when they lie inside it, so a database written in the container (`/app`) still
resolves on the host. All SQL is parameterized; table and column names are never interpolated
from input. The database holds paths and digests only, never pixel data (Req 11.6).

`enroll_template` is the only multi-statement write (Req 8.4, 8.6): one `BEGIN IMMEDIATE`
transaction computes the next version, writes `v<n>.xyt` atomically, deactivates the previous
active template and inserts the new active row. On any failure it rolls back, unlinks the new
file and re-raises, leaving the previous active template untouched.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import re
import sqlite3
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fpv.adapter import XytTemplate, write_xyt
from fpv.types import Finger, Hand, Identity, Status

if TYPE_CHECKING:
    from fpv.catalog import SampleRecord
    from fpv.config import PipelineConfig

__all__ = [
    "QualityAfterEntry",
    "Repository",
    "RunStart",
    "StageEvent",
    "TemplateMeta",
    "TemplateRow",
    "VerificationRow",
    "quality_after_entries",
    "utc_now_iso",
]

Operation = Literal["enroll", "verify"]
_MIGRATION_RE = re.compile(r"^(\d{4})_[A-Za-z0-9_]+\.sql$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DECISIONS = (Status.MATCH, Status.NO_MATCH)


def utc_now_iso() -> str:
    """UTC ISO-8601 timestamp with microseconds, e.g. `2026-10-24T20:49:00.123456+00:00`."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class QualityAfterEntry:
    """One attempt's `quality_after` as stored in `templates.quality_after` JSON (Req 8.2)."""

    attempt_no: int
    configuration_id: str
    score: int | None

    def to_json(self) -> dict[str, Any]:
        return {"attempt_no": self.attempt_no, "configuration_id": self.configuration_id, "score": self.score}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "QualityAfterEntry":
        return cls(int(obj["attempt_no"]), str(obj["configuration_id"]),
                   None if obj.get("score") is None else int(obj["score"]))


def quality_after_entries(attempts: Iterable[Any]) -> tuple[QualityAfterEntry, ...]:
    """Map `acquire.AttemptRecord`s (or anything with `attempt_no`, `configuration_id`,
    `quality_after`) to the per-attempt JSON entries, one per executed attempt."""
    return tuple(QualityAfterEntry(a.attempt_no, a.configuration_id, a.quality_after) for a in attempts)


@dataclass(frozen=True, slots=True)
class TemplateMeta:
    """Provenance supplied by the controller for a new template (Req 8.1, 8.2)."""

    source_sample_id: str
    quality_before: int
    quality_after: tuple[QualityAfterEntry, ...]
    scale_factor: float
    pipeline_version: str
    configuration_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality_after", tuple(self.quality_after))
        for e in self.quality_after:
            if not isinstance(e, QualityAfterEntry):
                raise TypeError(f"quality_after must hold QualityAfterEntry, got {type(e).__name__}")
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive number, got {self.scale_factor!r}")


@dataclass(frozen=True, slots=True)
class TemplateRow:
    template_id: str
    identity: Identity
    template_location: str
    source_sample_id: str
    quality_before: int
    quality_after: tuple[QualityAfterEntry, ...]
    minutiae_count: int
    scale_factor: float
    pipeline_version: str
    configuration_id: str
    template_version: int
    is_active: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class RunStart:
    """A new `runs` row (Req 11.2); status/reason/attempts/ended_at are set by `finish_run`."""

    run_id: str
    session_id: str
    presentation_no: int
    operation: Operation
    identity: Identity
    sample_id: str | None
    pipeline_version: str
    config_set_id: str
    started_at: str = field(default_factory=utc_now_iso)
    control_mode: str = "controller"
    orchestrator_vision: str = "off"


@dataclass(frozen=True, slots=True)
class StageEvent:
    """One `stage_events` row (Req 11.3)."""

    run_id: str
    seq: int
    stage: str
    started_at: str
    duration_ms: float
    outcome: str  # ok | rejected | error | skipped
    attempt_no: int | None = None
    configuration_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationRow:
    """One `verifications` row (Req 9.8)."""

    run_id: str
    template_id: str
    probe_location: str | Path
    score: int
    threshold: int
    decision: Status
    ground_truth_genuine: bool | None


# --------------------------------------------------------------------------- migrations


def _migrations() -> list[tuple[int, str, str]]:
    """(version, name, sql) for every packaged `migrations/NNNN_*.sql`, ordered by version."""
    out: list[tuple[int, str, str]] = []
    for entry in importlib.resources.files("fpv").joinpath("migrations").iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    out.sort()
    versions = [v for v, _, _ in out]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"migration versions must be 1..n without gaps, got {versions}")
    return out


def split_sql(script: str) -> list[str]:
    """Split a script into complete statements (comment/string aware via `complete_statement`).

    `executescript` would commit the surrounding transaction, so migrations run statement by
    statement inside `BEGIN IMMEDIATE` instead.
    """
    stmts: list[str] = []
    buf: list[str] = []
    for ch in script:
        buf.append(ch)
        if ch == ";" and sqlite3.complete_statement("".join(buf)):
            stmt = "".join(buf).strip()
            if stmt.strip(";").strip():
                stmts.append(stmt)
            buf = []
    tail = "".join(buf)
    if any(line.strip() and not line.strip().startswith("--") for line in tail.splitlines()):
        raise ValueError(f"incomplete SQL statement at end of migration: {tail.strip()[:80]!r}")
    return stmts


# --------------------------------------------------------------------------- repository


class Repository:
    def __init__(self, db_path: Path, templates_root: Path, probes_root: Path, *,
                 artifact_root: Path | None = None, clock: Callable[[], str] = utc_now_iso,
                 timeout_s: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.templates_root = Path(templates_root)
        self.probes_root = Path(probes_root)
        self.artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None
        self._clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit; transactions are explicit BEGIN IMMEDIATE ... COMMIT.
        self._conn = sqlite3.connect(self.db_path, timeout=timeout_s, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        mode = self._conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal" and str(self.db_path) != ":memory:":
            raise RuntimeError(f"could not enable WAL journal mode on {self.db_path} (got {mode})")
        self._conn.execute("PRAGMA synchronous = NORMAL")

    @classmethod
    def from_config(cls, conf: PipelineConfig, *, migrate: bool = True) -> "Repository":
        repo = cls(conf.path("db"), conf.path("templates"), conf.path("probes"), artifact_root=conf.root)
        if migrate:
            repo.migrate()
        return repo

    # ----------------------------------------------------------------- lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._conn.in_transaction:
            self._conn.execute("ROLLBACK")

    def migrate(self) -> list[int]:
        """Apply pending packaged migrations; idempotent. Returns the versions applied now."""
        applied_now: list[int] = []
        for version, name, sql in _migrations():
            self._begin()
            try:
                exists = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
                ).fetchone()
                done = exists and self._conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)).fetchone()
                if done:
                    self._conn.execute("COMMIT")
                    continue
                for stmt in split_sql(sql):
                    self._conn.execute(stmt)
                self._conn.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                                   (version, self._clock()))
                self._conn.execute("COMMIT")
            except BaseException as exc:
                self._rollback()
                if isinstance(exc, sqlite3.Error):
                    raise sqlite3.DatabaseError(f"migration {name} failed: {exc}") from exc
                raise
            applied_now.append(version)
        return applied_now

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    # ----------------------------------------------------------------- locations

    def _location(self, path: Path) -> str:
        p = Path(path).resolve()
        if self.artifact_root is not None:
            try:
                return p.relative_to(self.artifact_root).as_posix()
            except ValueError:
                pass
        return str(p)

    def resolve_location(self, location: str) -> Path:
        """Absolute path of a stored `template_location` / `probe_location`."""
        p = Path(location)
        if p.is_absolute() or self.artifact_root is None:
            return p
        return self.artifact_root / p

    def template_path(self, identity: Identity, version: int) -> Path:
        """`<templates_root>/<user>/<Hand>_<finger>/v<n>.xyt` (Req 8.1)."""
        return (self.templates_root / str(identity.user_id)
                / f"{identity.hand.value}_{identity.finger.value}" / f"v{int(version)}.xyt")

    def probe_path(self, run_id: str) -> Path:
        """`<probes_root>/<run_id>.xyt` (Req 9.1); `run_id` must be a plain file-name token."""
        if not isinstance(run_id, str) or not _SAFE_NAME_RE.match(run_id):
            raise ValueError(f"run_id {run_id!r} is not a safe artifact name")
        return self.probes_root / f"{run_id}.xyt"

    # ----------------------------------------------------------------- samples / runs / events

    def upsert_sample(self, rec: SampleRecord) -> None:
        """Insert or refresh a `samples` row (Req 11.1); `created_at` is kept from the first insert."""
        lab = rec.label
        ident = lab.identity if lab is not None else None
        self._conn.execute(
            """
            INSERT INTO samples (sample_id, source_path, sha256, subject_id, hand, finger, alteration,
                                 difficulty, split, excluded, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (sample_id) DO UPDATE SET
                source_path = excluded.source_path, sha256 = excluded.sha256,
                subject_id = excluded.subject_id, hand = excluded.hand, finger = excluded.finger,
                alteration = excluded.alteration, difficulty = excluded.difficulty,
                split = excluded.split, excluded = excluded.excluded,
                width = excluded.width, height = excluded.height
            """,
            (rec.sample_id, rec.source_path, rec.sha256,
             ident.user_id if ident else None,
             ident.hand.value if ident else None,
             ident.finger.value if ident else None,
             lab.alteration if lab else None,
             lab.difficulty if lab else None,
             rec.split, int(bool(rec.excluded)), int(rec.width), int(rec.height), self._clock()),
        )

    def start_run(self, run: RunStart) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, session_id, presentation_no, operation, claimed_user_id,
                              claimed_hand, claimed_finger, sample_id, control_mode,
                              orchestrator_vision, pipeline_version, config_set_id, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run.run_id, run.session_id, int(run.presentation_no), run.operation,
             run.identity.user_id, run.identity.hand.value, run.identity.finger.value, run.sample_id,
             run.control_mode, run.orchestrator_vision, run.pipeline_version, run.config_set_id,
             run.started_at),
        )

    def set_run_sample(self, run_id: str, sample_id: str) -> None:
        """Attach the sample once it has been loaded (a run may start before the image decodes)."""
        cur = self._conn.execute("UPDATE runs SET sample_id = ? WHERE run_id = ?", (sample_id, run_id))
        if cur.rowcount != 1:
            raise KeyError(f"unknown run_id {run_id!r}")

    def finish_run(self, run_id: str, status: Status, reason: str | None, attempts_used: int,
                   *, ended_at: str | None = None) -> None:
        cur = self._conn.execute(
            "UPDATE runs SET status = ?, status_reason = ?, attempts_used = ?, ended_at = ? WHERE run_id = ?",
            (Status(status).value, reason, int(attempts_used), ended_at or self._clock(), run_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"unknown run_id {run_id!r}")

    def log_stage_event(self, ev: StageEvent) -> None:
        details = json.dumps(dict(ev.details), sort_keys=True, separators=(",", ":"), default=str)
        self._conn.execute(
            """
            INSERT INTO stage_events (run_id, seq, stage, attempt_no, configuration_id, started_at,
                                      duration_ms, outcome, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ev.run_id, int(ev.seq), ev.stage, ev.attempt_no, ev.configuration_id, ev.started_at,
             float(ev.duration_ms), ev.outcome, details),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def stage_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM stage_events WHERE run_id = ? ORDER BY seq", (run_id,))
        return [{**dict(r), "details": json.loads(r["details_json"])} for r in rows]

    # ----------------------------------------------------------------- templates

    def enroll_template(self, identity: Identity, xyt: XytTemplate, meta: TemplateMeta) -> TemplateRow:
        """Persist `xyt` as the new active template of `identity` (Req 8.1-8.6)."""
        written: Path | None = None
        self._begin()
        try:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(template_version), 0) FROM templates "
                "WHERE user_id = ? AND hand = ? AND finger = ?",
                (identity.user_id, identity.hand.value, identity.finger.value),
            ).fetchone()
            version = int(row[0]) + 1
            path = self.template_path(identity, version)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_xyt(xyt, path)
            written = path
            self._after_artifact_write(path)
            new = TemplateRow(
                template_id=str(uuid.uuid4()),
                identity=identity,
                template_location=self._location(path),
                source_sample_id=meta.source_sample_id,
                quality_before=int(meta.quality_before),
                quality_after=meta.quality_after,
                minutiae_count=xyt.count,
                scale_factor=float(meta.scale_factor),
                pipeline_version=meta.pipeline_version,
                configuration_id=meta.configuration_id,
                template_version=version,
                is_active=True,
                created_at=self._clock(),
            )
            self._conn.execute(
                "UPDATE templates SET is_active = 0 "
                "WHERE user_id = ? AND hand = ? AND finger = ? AND is_active = 1",
                (identity.user_id, identity.hand.value, identity.finger.value),
            )
            self._conn.execute(
                """
                INSERT INTO templates (template_id, user_id, hand, finger, template_location,
                                       source_sample_id, quality_before, quality_after, minutiae_count,
                                       scale_factor, pipeline_version, configuration_id,
                                       template_version, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (new.template_id, identity.user_id, identity.hand.value, identity.finger.value,
                 new.template_location, new.source_sample_id, new.quality_before,
                 json.dumps([e.to_json() for e in new.quality_after], separators=(",", ":")),
                 new.minutiae_count, new.scale_factor, new.pipeline_version, new.configuration_id,
                 new.template_version, new.created_at),
            )
            self._conn.execute("COMMIT")
            return new
        except BaseException:
            self._rollback()
            if written is not None:
                written.unlink(missing_ok=True)
            raise

    def _after_artifact_write(self, path: Path) -> None:
        """Test seam: called inside the enrollment transaction right after the file write."""

    @staticmethod
    def _template_row(r: sqlite3.Row) -> TemplateRow:
        return TemplateRow(
            template_id=r["template_id"],
            identity=Identity(int(r["user_id"]), Hand(r["hand"]), Finger(r["finger"])),
            template_location=r["template_location"],
            source_sample_id=r["source_sample_id"],
            quality_before=int(r["quality_before"]),
            quality_after=tuple(QualityAfterEntry.from_json(o) for o in json.loads(r["quality_after"])),
            minutiae_count=int(r["minutiae_count"]),
            scale_factor=float(r["scale_factor"]),
            pipeline_version=r["pipeline_version"],
            configuration_id=r["configuration_id"],
            template_version=int(r["template_version"]),
            is_active=bool(r["is_active"]),
            created_at=r["created_at"],
        )

    def get_active_template(self, identity: Identity) -> TemplateRow | None:
        r = self._conn.execute(
            "SELECT * FROM templates WHERE user_id = ? AND hand = ? AND finger = ? AND is_active = 1",
            (identity.user_id, identity.hand.value, identity.finger.value),
        ).fetchone()
        return self._template_row(r) if r is not None else None

    def list_templates(self, identity: Identity) -> list[TemplateRow]:
        """Every version for `identity`, oldest first (prior versions are retained, Req 8.4)."""
        rows = self._conn.execute(
            "SELECT * FROM templates WHERE user_id = ? AND hand = ? AND finger = ? ORDER BY template_version",
            (identity.user_id, identity.hand.value, identity.finger.value),
        )
        return [self._template_row(r) for r in rows]

    # ----------------------------------------------------------------- probes / verifications

    def write_probe(self, run_id: str, xyt: XytTemplate) -> Path:
        """Store a verification probe at `<probes_root>/<run_id>.xyt` (atomic write)."""
        path = self.probe_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_xyt(xyt, path)
        return path

    def record_verification(self, v: VerificationRow) -> None:
        decision = Status(v.decision)
        if decision not in _DECISIONS:
            raise ValueError(f"verification decision must be MATCH or NO_MATCH, got {decision.value}")
        location = v.probe_location if isinstance(v.probe_location, str) else self._location(v.probe_location)
        gt = None if v.ground_truth_genuine is None else int(bool(v.ground_truth_genuine))
        self._conn.execute(
            """
            INSERT INTO verifications (run_id, template_id, probe_location, score, threshold, decision,
                                       ground_truth_genuine)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (v.run_id, v.template_id, location, int(v.score), int(v.threshold), decision.value, gt),
        )

    def get_verification(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM verifications WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None


# --------------------------------------------------------------------------- listing / demo


def format_templates(rows: Sequence[TemplateRow]) -> list[str]:
    """Table lines for `fpv db templates`."""
    header = ("version", "active", "template_id", "minutiae", "q_before", "configuration_id",
              "created_at", "location")
    table = [header] + [
        (f"v{r.template_version}", "yes" if r.is_active else "no", r.template_id, str(r.minutiae_count),
         str(r.quality_before), r.configuration_id, r.created_at, r.template_location)
        for r in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table]


def _demo_template(seed: int, n: int = 14) -> XytTemplate:
    from fpv.adapter import XytMinutia

    pts: dict[tuple[int, int], XytMinutia] = {}
    k = 0
    while len(pts) < n:
        x, y = (17 * k + 5 * seed) % 96, (29 * k + 3 * seed) % 103
        pts[(x, y)] = XytMinutia(x, y, (37 * k + 11 * seed) % 360, 50 + (k % 50))
        k += 1
    return XytTemplate(tuple(sorted(pts.values(), key=lambda m: (m.y, m.x, m.theta))))


def run_demo(base: Path, out: Callable[[str], None] = print) -> Repository:
    """Enroll a fake template twice for user 1 Left index in a throwaway store under `base`."""
    from fpv.catalog import SampleRecord, make_sample_id

    repo = Repository(base / "db" / "fpv.sqlite", base / "templates", base / "probes", artifact_root=base)
    repo.migrate()
    identity = Identity.parse(1, "Left", "index")
    sha = "0" * 64
    src = "Real/1__M_Left_index_finger.BMP (demo, not read)"
    sample = SampleRecord(make_sample_id(sha, src), src, sha, 96, 103, None, None, False)
    repo.upsert_sample(sample)
    for seed in (1, 2):
        row = repo.enroll_template(identity, _demo_template(seed), TemplateMeta(
            source_sample_id=sample.sample_id, quality_before=40 + seed,
            quality_after=(QualityAfterEntry(1, "cfg_demo00000001", 55 + seed),),
            scale_factor=3.0, pipeline_version="demo", configuration_id="cfg_demo00000001"))
        out(f"enrolled {identity}: template v{row.template_version} ({row.template_id}) -> "
            f"{repo.resolve_location(row.template_location)}")
    for line in format_templates(repo.list_templates(identity)):
        out(line)
    return repo


def main(argv: Sequence[str] | None = None) -> int:
    import tempfile

    ap = argparse.ArgumentParser(prog="python -m fpv.storage", description="fpv storage layer")
    ap.add_argument("--demo", action="store_true", help="enroll a fake template twice in a temporary store")
    ap.add_argument("--dir", type=Path, default=None,
                    help="demo store directory (default: a new temporary directory)")
    args = ap.parse_args(argv)
    if not args.demo:
        ap.print_help()
        return 2
    base = args.dir or Path(tempfile.mkdtemp(prefix="fpv-storage-demo-"))
    base.mkdir(parents=True, exist_ok=True)
    with run_demo(base) as repo:
        db = repo.db_path
    print(f"\nlist it with:  fpv db templates --db {db} --user 1 --hand Left --finger index")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

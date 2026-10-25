"""Unit tests for `fpv.storage` on a temporary DB and artifact directory (Req 8, 9.1, 9.8, 11)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpv import storage as storage_mod
from fpv.acquire import AttemptRecord
from fpv.adapter import XytMinutia, XytTemplate, read_xyt
from fpv.catalog import SampleLabel, SampleRecord, make_sample_id
from fpv.cli import EXIT_OK, main
from fpv.storage import (
    QualityAfterEntry,
    Repository,
    RunStart,
    StageEvent,
    TemplateMeta,
    VerificationRow,
    quality_after_entries,
    split_sql,
)
from fpv.types import Identity, Status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"
ID = Identity.parse(1, "Left", "index")


def _xyt(n: int = 12, offset: int = 0) -> XytTemplate:
    return XytTemplate(tuple(XytMinutia(5 + 7 * i, 3 + offset + 5 * i, (37 * i) % 360, 50 + i)
                             for i in range(n)))


def _sample(path: str = "Real/1__M_Left_index_finger.BMP", labeled: bool = True) -> SampleRecord:
    sha = "ab" * 32
    label = SampleLabel(ID, "M", "Real", "Real") if labeled else None
    return SampleRecord(make_sample_id(sha, path), path, sha, 96, 103, label, "train" if labeled else None, False)


ATTEMPTS = (
    AttemptRecord(1, "cfg_aaaaaaaaaaaa", None, 8, 8),
    AttemptRecord(2, "cfg_bbbbbbbbbbbb", 57, 30, 24),
)


def _meta(sample_id: str, quality_before: int = 41) -> TemplateMeta:
    return TemplateMeta(source_sample_id=sample_id, quality_before=quality_before,
                        quality_after=quality_after_entries(ATTEMPTS), scale_factor=3.0,
                        pipeline_version="0.1.0+gabc1234", configuration_id="cfg_bbbbbbbbbbbb")


@pytest.fixture
def repo(tmp_path: Path):
    r = Repository(tmp_path / "db" / "fpv.sqlite", tmp_path / "templates", tmp_path / "probes",
                   artifact_root=tmp_path)
    r.migrate()
    yield r
    r.close()


@pytest.fixture
def sample(repo: Repository) -> SampleRecord:
    s = _sample()
    repo.upsert_sample(s)
    return s


def _run(repo: Repository, run_id: str, sample_id: str | None, op: str = "verify") -> RunStart:
    run = RunStart(run_id=run_id, session_id="sess-1", presentation_no=1, operation=op, identity=ID,
                   sample_id=sample_id, pipeline_version="0.1.0+gabc1234", config_set_id="cs_000000000000")
    repo.start_run(run)
    return run


# --------------------------------------------------------------------------- connection / migrations


def test_connection_pragmas(repo: Repository) -> None:
    assert repo.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert repo.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrations_are_idempotent(tmp_path: Path, repo: Repository) -> None:
    assert repo.schema_version() == 1
    assert repo.migrate() == []
    # A second connection to the same file applies nothing either.
    with Repository(repo.db_path, tmp_path / "templates", tmp_path / "probes") as other:
        assert other.migrate() == []
    rows = repo.connection.execute("SELECT version FROM schema_migrations").fetchall()
    assert [r[0] for r in rows] == [1]


def test_fresh_migrate_reports_applied(tmp_path: Path) -> None:
    with Repository(tmp_path / "x.sqlite", tmp_path / "t", tmp_path / "p") as r:
        assert r.migrate() == [1]
        assert r.migrate() == []


def test_split_sql_ignores_semicolons_in_comments_and_strings() -> None:
    stmts = split_sql("-- a; b\nCREATE TABLE t (x TEXT DEFAULT 'a;b');\n\nSELECT 1; -- tail\n")
    assert stmts == ["-- a; b\nCREATE TABLE t (x TEXT DEFAULT 'a;b');", "SELECT 1;"]
    with pytest.raises(ValueError):
        split_sql("SELECT 1; SELECT 2")


def test_no_pixel_data_columns(repo: Repository) -> None:
    tables = [r[0] for r in repo.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
    assert set(tables) >= {"samples", "templates", "runs", "stage_events", "verifications"}
    for t in tables:
        for col in repo.connection.execute("SELECT name, type FROM pragma_table_info(?)", (t,)):
            name, typ = col[0].lower(), (col[1] or "").upper()
            assert "BLOB" not in typ, f"{t}.{name} is a BLOB column"
            assert not any(k in name for k in ("pixel", "image", "png", "bitmap")), f"{t}.{name}"


# --------------------------------------------------------------------------- templates


def test_enroll_twice_versions_and_active(repo: Repository, sample: SampleRecord, tmp_path: Path) -> None:
    v1 = repo.enroll_template(ID, _xyt(12), _meta(sample.sample_id))
    v2 = repo.enroll_template(ID, _xyt(14, offset=1), _meta(sample.sample_id, 45))
    assert (v1.template_version, v2.template_version) == (1, 2)

    rows = repo.list_templates(ID)
    assert [(r.template_version, r.is_active) for r in rows] == [(1, False), (2, True)]
    assert repo.get_active_template(ID).template_id == v2.template_id

    p1 = tmp_path / "templates" / "1" / "Left_index" / "v1.xyt"
    p2 = tmp_path / "templates" / "1" / "Left_index" / "v2.xyt"
    assert rows[0].template_location == "templates/1/Left_index/v1.xyt"
    assert repo.resolve_location(rows[0].template_location) == p1
    assert p1.is_file() and p2.is_file()
    assert read_xyt(p1) == _xyt(12)
    assert read_xyt(p2) == _xyt(14, offset=1)


def test_second_active_insert_violates_partial_unique_index(repo: Repository, sample: SampleRecord) -> None:
    repo.enroll_template(ID, _xyt(), _meta(sample.sample_id))
    with pytest.raises(sqlite3.IntegrityError, match="ux_templates_one_active|UNIQUE"):
        repo.connection.execute(
            """INSERT INTO templates (template_id, user_id, hand, finger, template_location,
                   source_sample_id, quality_before, quality_after, minutiae_count, scale_factor,
                   pipeline_version, configuration_id, template_version, is_active, created_at)
               VALUES ('t-x', 1, 'Left', 'index', 'x', ?, 1, '[]', 12, 1.0, 'v', 'c', 99, 1, 'now')""",
            (sample.sample_id,),
        )
    # An inactive row for the same identity is allowed: the index is partial.
    repo.connection.execute(
        """INSERT INTO templates (template_id, user_id, hand, finger, template_location,
               source_sample_id, quality_before, quality_after, minutiae_count, scale_factor,
               pipeline_version, configuration_id, template_version, is_active, created_at)
           VALUES ('t-y', 1, 'Left', 'index', 'x', ?, 1, '[]', 12, 1.0, 'v', 'c', 99, 0, 'now')""",
        (sample.sample_id,),
    )


def test_get_active_template_unknown_identity(repo: Repository, sample: SampleRecord) -> None:
    repo.enroll_template(ID, _xyt(), _meta(sample.sample_id))
    assert repo.get_active_template(Identity.parse(2, "Left", "index")) is None
    assert repo.get_active_template(Identity.parse(1, "Right", "index")) is None
    assert repo.list_templates(Identity.parse(600, "Right", "thumb")) == []


def test_template_metadata_round_trips(repo: Repository, sample: SampleRecord) -> None:
    created = repo.enroll_template(ID, _xyt(13), _meta(sample.sample_id))
    got = repo.get_active_template(ID)
    assert got == created
    assert got.identity == ID
    assert got.source_sample_id == sample.sample_id
    assert got.quality_before == 41
    assert got.minutiae_count == 13
    assert got.scale_factor == 3.0
    assert got.pipeline_version == "0.1.0+gabc1234"
    assert got.configuration_id == "cfg_bbbbbbbbbbbb"
    assert got.is_active is True and got.template_version == 1
    # UUID4 template_id and UTC ISO-8601 created_at (Req 8.3).
    import uuid
    from datetime import datetime, timedelta
    assert uuid.UUID(got.template_id).version == 4
    assert datetime.fromisoformat(got.created_at).utcoffset() == timedelta(0)
    # quality_after: JSON list with one entry per executed attempt (Req 8.2).
    assert got.quality_after == (QualityAfterEntry(1, "cfg_aaaaaaaaaaaa", None),
                                 QualityAfterEntry(2, "cfg_bbbbbbbbbbbb", 57))
    raw = repo.connection.execute("SELECT quality_after FROM templates").fetchone()[0]
    assert json.loads(raw) == [{"attempt_no": 1, "configuration_id": "cfg_aaaaaaaaaaaa", "score": None},
                               {"attempt_no": 2, "configuration_id": "cfg_bbbbbbbbbbbb", "score": 57}]


def test_failure_after_file_write_rolls_back(repo: Repository, sample: SampleRecord,
                                             tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1 = repo.enroll_template(ID, _xyt(), _meta(sample.sample_id))
    v2_path = tmp_path / "templates" / "1" / "Left_index" / "v2.xyt"
    seen: list[bool] = []

    def boom(path: Path) -> None:
        seen.append(path.is_file())  # the file really was written before the failure
        raise OSError("injected failure after artifact write")

    monkeypatch.setattr(repo, "_after_artifact_write", boom)
    with pytest.raises(OSError, match="injected"):
        repo.enroll_template(ID, _xyt(15), _meta(sample.sample_id))
    assert seen == [True]
    assert not v2_path.exists()
    assert [r.template_version for r in repo.list_templates(ID)] == [1]
    assert repo.get_active_template(ID).template_id == v1.template_id
    assert not repo.connection.in_transaction
    # The store is still usable and the next version is 2 again.
    monkeypatch.undo()
    assert repo.enroll_template(ID, _xyt(15), _meta(sample.sample_id)).template_version == 2


def test_insert_failure_after_file_write_rolls_back(repo: Repository, sample: SampleRecord, tmp_path: Path) -> None:
    v1 = repo.enroll_template(ID, _xyt(), _meta(sample.sample_id))
    with pytest.raises(sqlite3.IntegrityError):  # unknown source sample -> FK violation on INSERT
        repo.enroll_template(ID, _xyt(15), _meta("s_does_not_exist"))
    assert not (tmp_path / "templates" / "1" / "Left_index" / "v2.xyt").exists()
    assert [(r.template_version, r.is_active) for r in repo.list_templates(ID)] == [(1, True)]
    assert repo.get_active_template(ID).template_id == v1.template_id


# --------------------------------------------------------------------------- samples / runs / events


def test_upsert_sample_labeled_and_unlabeled(repo: Repository) -> None:
    s = _sample()
    repo.upsert_sample(s)
    repo.upsert_sample(s)  # idempotent
    u = _sample("/tmp/elsewhere.png", labeled=False)
    repo.upsert_sample(u)
    rows = {r["sample_id"]: dict(r) for r in repo.connection.execute("SELECT * FROM samples")}
    assert len(rows) == 2
    assert (rows[s.sample_id]["subject_id"], rows[s.sample_id]["hand"], rows[s.sample_id]["finger"],
            rows[s.sample_id]["split"]) == (1, "Left", "index", "train")
    assert rows[u.sample_id]["subject_id"] is None and rows[u.sample_id]["alteration"] is None


def test_run_lifecycle_and_stage_events(repo: Repository, sample: SampleRecord) -> None:
    _run(repo, "run-1", sample.sample_id, op="enroll")
    repo.log_stage_event(StageEvent("run-1", 0, "quality_before", "2026-01-01T00:00:00+00:00", 3.5, "ok",
                                    details={"score": 41}))
    repo.log_stage_event(StageEvent("run-1", 1, "preprocess", "2026-01-01T00:00:01+00:00", 120.0, "ok",
                                    attempt_no=1, configuration_id="cfg_aaaaaaaaaaaa"))
    repo.finish_run("run-1", Status.ENROLLED, None, 1)
    run = repo.get_run("run-1")
    assert run["status"] == "ENROLLED" and run["attempts_used"] == 1 and run["ended_at"]
    assert run["control_mode"] == "controller" and run["orchestrator_vision"] == "off"
    assert run["config_set_id"] == "cs_000000000000"
    evs = repo.stage_events("run-1")
    assert [(e["seq"], e["stage"], e["attempt_no"]) for e in evs] == [(0, "quality_before", None),
                                                                     (1, "preprocess", 1)]
    assert evs[0]["details"] == {"score": 41}
    with pytest.raises(sqlite3.IntegrityError):  # UNIQUE (run_id, seq)
        repo.log_stage_event(StageEvent("run-1", 1, "extract", "t", 1.0, "ok"))
    with pytest.raises(KeyError):
        repo.finish_run("nope", Status.ERROR, "x", 0)


def test_stage_event_for_unknown_run_fails_fk(repo: Repository) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        repo.log_stage_event(StageEvent("no-such-run", 0, "load", "2026-01-01T00:00:00+00:00", 1.0, "ok"))


def test_probe_and_verification(repo: Repository, sample: SampleRecord, tmp_path: Path) -> None:
    tpl = repo.enroll_template(ID, _xyt(), _meta(sample.sample_id))
    _run(repo, "run-v", sample.sample_id)
    path = repo.write_probe("run-v", _xyt(11))
    assert path == tmp_path / "probes" / "run-v.xyt" and read_xyt(path) == _xyt(11)
    repo.record_verification(VerificationRow("run-v", tpl.template_id, path, 55, 40, Status.MATCH, True))
    v = repo.get_verification("run-v")
    assert v == {"run_id": "run-v", "template_id": tpl.template_id, "probe_location": "probes/run-v.xyt",
                 "score": 55, "threshold": 40, "decision": "MATCH", "ground_truth_genuine": 1}

    _run(repo, "run-w", None)
    repo.record_verification(VerificationRow("run-w", tpl.template_id, "probes/run-w.xyt", 3, 40,
                                             Status.NO_MATCH, None))
    assert repo.get_verification("run-w")["ground_truth_genuine"] is None
    with pytest.raises(ValueError):
        repo.record_verification(VerificationRow("run-w", tpl.template_id, "p", 1, 40, Status.ERROR, None))
    with pytest.raises(ValueError):
        repo.write_probe("../escape", _xyt())


# --------------------------------------------------------------------------- CLI / demo


def test_db_templates_cli_lists_versions(tmp_path: Path) -> None:
    repo = storage_mod.run_demo(tmp_path, out=lambda s: None)
    repo.close()
    lines: list[str] = []
    rc = main(["--config", str(CONFIG), "db", "templates", "--db", str(tmp_path / "db" / "fpv.sqlite"),
               "--user", "1", "--hand", "left", "--finger", "INDEX"], out=lines.append, err=lambda s: None)
    assert rc == EXIT_OK
    rows = [line.split() for line in lines if line[:2] in ("v1", "v2")]
    assert [(r[0], r[1]) for r in rows] == [("v1", "no"), ("v2", "yes")]


def test_db_command_is_hidden_from_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert " db " not in capsys.readouterr().out

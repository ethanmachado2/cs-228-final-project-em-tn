"""Pipeline controller end to end on SOCOFing with the real toolchain. Run in the container."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpv import config as C
from fpv.catalog import Catalog, dataset_key, parse_label
from fpv.pipeline import PipelineController, main
from fpv.storage import Repository
from fpv.types import Status

pytestmark = [pytest.mark.toolchain, pytest.mark.dataset]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "pipeline.v1.yaml"
RAW = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(CONFIG, root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def catalog(conf) -> Catalog:
    return Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))


def left_index_pair(catalog: Catalog, start: int) -> tuple[Path, Path]:
    """(Real, Altered-Easy CR) Left index of the first subject >= `start` with neither excluded."""
    for subject in range(start, 601):
        for real in sorted((RAW / "Real").glob(f"{subject}__*_Left_index_finger.BMP")):
            cr = RAW / "Altered" / "Altered-Easy" / f"{real.stem}_CR.BMP"
            if cr.is_file() and not any(catalog.is_excluded(dataset_key(p, parse_label(p))) for p in (real, cr)):
                return real, cr
    raise AssertionError("no non-excluded Left index pair found")


def test_enroll_then_verify_genuine_and_impostor(tmp_path, conf, catalog):
    real, cr = left_index_pair(catalog, 1)
    other_real, _ = left_index_pair(catalog, parse_label(real).identity.user_id + 1)
    identity = parse_label(real).identity

    with Repository(tmp_path / "db" / "fpv.sqlite", tmp_path / "templates", tmp_path / "probes",
                    artifact_root=tmp_path) as repo:
        repo.migrate()
        ctl = PipelineController(conf, repo, catalog)
        enrolled = ctl.enroll(real, identity, session_id="it")
        assert enrolled.status is Status.ENROLLED, enrolled
        assert repo.resolve_location(repo.get_active_template(identity).template_location).is_file()

        genuine = ctl.verify(cr, identity, session_id="it")
        assert genuine.status is Status.MATCH, genuine
        assert genuine.ground_truth_genuine is True

        impostor = ctl.verify(other_real, identity, session_id="it")
        assert impostor.status is Status.NO_MATCH, impostor
        assert impostor.ground_truth_genuine is False
        assert genuine.score > impostor.score

        for out in (enrolled, genuine, impostor):
            run = repo.get_run(out.run_id)
            assert run["ended_at"] and run["config_set_id"] == conf.config_set_id
            assert repo.stage_events(out.run_id)[0]["stage"] == "load"
        assert [e["stage"] for e in repo.stage_events(genuine.run_id)][-4:] == ["persist", "lookup", "match",
                                                                                "decide"]


def test_demo_prints_match(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(PROJECT_ROOT)
    code = main(["--demo", "--config", str(CONFIG), "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.strip().splitlines()[-1] == "MATCH"
    assert "decide" in out and " ms" in out

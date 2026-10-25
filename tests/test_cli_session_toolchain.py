"""`fpv enroll` / `fpv verify` end to end on SOCOFing with the real toolchain (Task 10 demo flow).

Uses a temporary copy of the config whose db/templates/probes paths point into `tmp_path`, so
nothing is written to the real `data/` store. Run in the container.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fpv import cli
from fpv import config as C
from fpv.catalog import Catalog, parse_label

from test_pipeline_toolchain import CONFIG, PROJECT_ROOT, left_index_pair

pytestmark = [pytest.mark.toolchain, pytest.mark.dataset]


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key in ("raw", "splits", "exclusions", "reports"):
        raw["paths"][key] = str(PROJECT_ROOT / raw["paths"][key])
    raw["paths"].update(db=str(tmp_path / "db" / "fpv.sqlite"), templates=str(tmp_path / "templates"),
                        probes=str(tmp_path / "probes"))
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_enroll_interactive_then_verify_non_interactive(temp_config, tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    conf = C.load(temp_config, root=PROJECT_ROOT)
    catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    real, cr = left_index_pair(catalog, 1)
    identity = parse_label(real).identity
    other_real, _ = left_index_pair(catalog, identity.user_id + 1)

    def run(argv, inputs=()):
        queue, out, err = list(inputs), [], []
        code = cli.main(["--config", str(temp_config), *argv], out=out.append, err=err.append,
                        input_fn=lambda prompt: queue.pop(0))
        assert not queue
        return code, out, err

    code, out, err = run(["enroll"], [str(real), str(identity.user_id), identity.hand.value,
                                      identity.finger.value, "y"])
    assert (code, out[-1]) == (cli.EXIT_OK, "Enrollment successful (template v1)"), out
    assert f"({conf.config_set_id})" in err[0]
    assert (tmp_path / "templates").is_dir()

    claim = ["--user", str(identity.user_id), "--hand", identity.hand.value, "--finger", identity.finger.value,
             "--yes"]
    code, out, _ = run(["verify", "--image", str(cr), *claim])
    assert (code, out[-1]) == (cli.EXIT_OK, "Match"), out
    code, out, _ = run(["verify", "--image", str(other_real), *claim])
    assert (code, out[-1]) == (cli.EXIT_NO_MATCH, "No match"), out

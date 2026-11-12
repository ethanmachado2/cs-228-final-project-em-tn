"""Live checks against `docker compose up -d`, run from the harness container (Task 5, Req 6).

    docker compose run --rm harness pytest -m live tests/test_agents_live.py

Skipped when the controller agent is not reachable (host runs, the `fpv` service).
"""

from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("a2a")

from fpv.agents.serve import AGENTS, agent_url  # noqa: E402

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parent.parent
TELEMETRY = Path(os.environ.get("FPV_TELEMETRY_DIR", ROOT / "data" / "telemetry"))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module", autouse=True)
def live():
    try:
        _get(agent_url("controller") + ".well-known/agent-card.json")
    except OSError as exc:
        pytest.skip(f"agents not reachable: {exc}")


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_every_agent_serves_its_card_and_is_healthy(name):
    card = _get(agent_url(name) + ".well-known/agent-card.json")
    assert card["name"] == name and [s["id"] for s in card["skills"]] == list(AGENTS[name].skills)
    with urllib.request.urlopen(agent_url(name) + "healthz", timeout=5) as r:
        assert r.status == 200


@pytest.mark.parametrize("target", [("1.1.1.1", 443), ("pypi.org", 443)])
def test_no_internet_from_the_a2a_network(target):
    with pytest.raises(OSError):
        socket.create_connection(target, timeout=3).close()


@pytest.mark.parametrize("name", [n for n in sorted(AGENTS) if n != "controller"])
def test_startup_warmup_is_recorded_and_flagged(name):
    hops = [json.loads(x) for x in (TELEMETRY / f"{name}.jsonl").read_text().splitlines()]
    own = [h for h in hops if h.get("kind") == "hop" and h["http_method"] == "POST" and h["peer"] == f"{name}-warmup"]
    assert {h["skill"] for h in own} == set(AGENTS[name].skills)
    assert all(h["warmup"] and h["final_state"] == "TASK_STATE_COMPLETED" for h in own)


@pytest.mark.dataset
def test_verify_through_the_controller():
    from fpv.agents.client import present

    raw = ROOT / "data" / "raw"
    state, d, _ = present(agent_url("controller"), "enroll", (raw / "Real/1__M_Left_index_finger.BMP").read_bytes(),
                          1, "Left", "index")
    assert (state, d["status"]) == ("TASK_STATE_COMPLETED", "ENROLLED")
    state, d, _ = present(agent_url("controller"), "verify",
                          (raw / "Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP").read_bytes(), 1, "Left", "index")
    assert (state, d["status"], d["score"]) == ("TASK_STATE_COMPLETED", "MATCH", 296)

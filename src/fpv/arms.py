"""Experiment arms and the A2A harness (spec a2a-agents-telemetry, Req 7).

Arms (SOCOFing has one Real impression per finger):

- `A`  Real image against its own enrollment (ceiling; the probe is the enrolled image itself)
- `B`  Real image of another subject with the same hand and finger, claiming an enrolled identity
       (zero-effort impostor, the main control)
- `C`  Altered image claiming its own identity (altered genuine), from `evaluate.generate_trials`
- `C'` Altered image claiming another enrolled identity (altered impostor), from the same trials

`build_arm_plan` is deterministic for (catalog, split, seed, k, per_arm, subjects): each arm is
sampled down to `per_arm` with its own seeded RNG and all arms are interleaved by one seeded
permutation (Req 7.2). `run_a2a` enrolls the needed identities through the controller, then sends
the trials one at a time, flagging the first `warmup` as warm-up (Req 7.3). It writes one label per
trial to `labels.jsonl` before sending it; only an opaque random `trial_id` goes on the wire (Req 7.4).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fpv.evaluate import Enrollment, _stable_int, generate_trials
from fpv.types import Identity

if TYPE_CHECKING:
    from fpv.catalog import Catalog
    from fpv.config import PipelineConfig

ARMS = ("A", "B", "C", "C'")
DEFAULT_PER_ARM = 300
DEFAULT_WARMUP = 20


@dataclass(frozen=True, slots=True)
class ArmTrial:
    arm: str
    probe: str               # dataset key relative to the raw root
    probe_identity: Identity
    claim: Identity
    genuine: bool            # probe subject == claimed subject
    alteration: str | None   # CR | Obl | Zcut for C / C'
    difficulty: str | None   # Easy | Medium | Hard for C / C'


@dataclass(frozen=True, slots=True)
class ArmPlan:
    split: str
    seed: int
    enrollments: tuple[Enrollment, ...]  # only the identities some trial claims
    trials: tuple[ArmTrial, ...]         # interleaved send order


def build_arm_plan(catalog: Catalog, split: str, seed: int, k: int, per_arm: int,
                   subjects: int | None = None) -> ArmPlan:
    if per_arm < 1:
        raise ValueError(f"per_arm must be at least 1, got {per_arm}")
    plan = generate_trials(catalog, split, seed, k, subjects)
    by_arm: dict[str, list[ArmTrial]] = {a: [] for a in ARMS}
    by_position: dict[tuple[str, str], list[Identity]] = {}
    for e in plan.enrollments:
        by_position.setdefault((e.identity.hand.value, e.identity.finger.value), []).append(e.identity)
    for e in plan.enrollments:
        by_arm["A"].append(ArmTrial("A", e.sample, e.identity, e.identity, True, None, None))
        others = sorted((i for i in by_position[(e.identity.hand.value, e.identity.finger.value)]
                         if i.user_id != e.identity.user_id), key=lambda i: i.user_id)
        if others:
            rng = np.random.default_rng([seed, _stable_int(e.sample), 1])
            claim = others[int(rng.integers(len(others)))]
            by_arm["B"].append(ArmTrial("B", e.sample, e.identity, claim, False, None, None))
    for t in plan.trials:
        by_arm["C" if t.genuine else "C'"].append(
            ArmTrial("C" if t.genuine else "C'", t.probe, t.probe_identity, t.claim, t.genuine, t.alteration,
                     t.difficulty))
    chosen: list[ArmTrial] = []
    for n, arm in enumerate(ARMS):
        items = by_arm[arm]
        if len(items) > per_arm:
            idx = sorted(int(i) for i in np.random.default_rng([seed, 7, n]).choice(len(items), per_arm, replace=False))
            items = [items[i] for i in idx]
        chosen += items
    order = np.random.default_rng([seed, 99]).permutation(len(chosen))
    trials = tuple(chosen[int(i)] for i in order)
    needed = {t.claim for t in trials}
    return ArmPlan(split, seed, tuple(e for e in plan.enrollments if e.identity in needed), trials)


def label_record(trial_id: str, t: ArmTrial, plan: ArmPlan, warmup: bool) -> dict[str, Any]:
    return {"trial_id": trial_id, "arm": t.arm, "genuine": t.genuine, "alteration": t.alteration,
            "difficulty": t.difficulty, "subject": t.probe_identity.user_id, "probe": t.probe,
            "claim": {"user_id": t.claim.user_id, "hand": t.claim.hand.value, "finger": t.claim.finger.value},
            "split": plan.split, "seed": plan.seed, "warmup": warmup}


# --------------------------------------------------------------------------- A2A harness


def run_a2a(conf: PipelineConfig, plan: ArmPlan, *, labels_path: Path, url: str | None = None,
            warmup: int = DEFAULT_WARMUP, control_mode: str = "controller",
            out: Callable[[str], None] = print, client: Any = None) -> Counter:
    """Enroll, then send every trial to the controller agent; returns (arm, status) counts."""
    from fpv.agents.client import A2AClient, result_payload
    from fpv.agents.controller import sample_payload
    from fpv.agents.serve import agent_url
    from fpv.telemetry import writer_from_env

    raw = conf.path("raw")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    client = client or A2AClient(url or agent_url("controller"), "controller", agent="harness",
                                 writer=writer_from_env("harness"))
    counts: Counter = Counter()

    async def send(skill: str, key: str, claim: Identity, **meta: Any) -> tuple[str, dict[str, Any]]:
        p = sample_payload((raw / key).read_bytes(), claim.user_id, claim.hand.value, claim.finger.value)
        state, payload = result_payload(await client.call(p, {"skill": skill, "control_mode": control_mode, **meta}))
        return state, payload.data

    async def go() -> None:
        with open(labels_path, "a", encoding="utf-8") as labels:
            for e in plan.enrollments:
                tid = f"e{uuid.uuid4().hex[:15]}"
                labels.write(json.dumps({"trial_id": tid, "arm": "enroll", "probe": e.sample,
                                         "subject": e.identity.user_id, "split": plan.split, "seed": plan.seed,
                                         "warmup": True}) + "\n")
                labels.flush()
                _, d = await send("enroll", e.sample, e.identity, trial_id=tid, warmup=True)
                counts[("enroll", d.get("status") or d.get("error_code"))] += 1
            out(f"enrolled {len(plan.enrollments)} identities: {dict(counts)}")
            for i, t in enumerate(plan.trials):
                tid = f"t{uuid.uuid4().hex[:15]}"
                is_warm = i < warmup
                labels.write(json.dumps(label_record(tid, t, plan, is_warm)) + "\n")
                labels.flush()
                state, d = await send("verify", t.probe, t.claim, trial_id=tid, warmup=is_warm)
                counts[(t.arm, d.get("status") or d.get("error_code") or state)] += 1
                if (i + 1) % 25 == 0 or i + 1 == len(plan.trials):
                    out(f"{i + 1}/{len(plan.trials)} trials")
        await client.aclose()

    asyncio.run(go())
    return counts


# --------------------------------------------------------------------------- scripted llm_routed arm


def pick_llm_trials(labels: list[dict[str, Any]], per_arm: int, seed: int) -> list[dict[str, Any]]:
    """`per_arm` non-warm-up harness trials per arm (seeded), interleaved (Req 10.1)."""
    chosen: list[dict[str, Any]] = []
    for n, arm in enumerate(ARMS):
        items = [r for r in labels if r.get("arm") == arm and not r.get("warmup")]
        if len(items) > per_arm:
            idx = sorted(int(i) for i in np.random.default_rng([seed, 11, n]).choice(len(items), per_arm, replace=False))
            items = [items[i] for i in idx]
        chosen += items
    return [chosen[int(i)] for i in np.random.default_rng([seed, 13]).permutation(len(chosen))]


def _adk_post(base: str) -> Callable[[str, dict[str, Any]], Any]:
    import urllib.request

    def post(path: str, body: dict[str, Any]) -> Any:
        req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read() or b"null")
    return post


def scripted_trial(post: Callable[[str, dict[str, Any]], Any], label: dict[str, Any], image: bytes,
                   app: str = "fpv_orchestrator") -> dict[str, Any]:
    """One scripted conversation through the orchestrator's ADK server: claim + upload, then "yes"."""
    import base64

    claim = label["claim"]
    session = post(f"/apps/{app}/users/harness/sessions", {"state": {"trial_id": label["trial_id"]}})
    turns = [[{"text": f"I want to verify my fingerprint. I claim user {claim['user_id']}, "
                       f"{claim['hand']} {claim['finger']} finger. Here is the image."},
              {"inline_data": {"mime_type": "image/bmp", "data": base64.b64encode(image).decode("ascii")}}],
             [{"text": "yes"}]]
    rec: dict[str, Any] = {"trial_id": label["trial_id"], "arm": label["arm"], "session_id": session["id"],
                           "input_tokens": 0, "output_tokens": 0, "model_calls": 0, "tool_calls": [],
                           "status": None, "reply": None}
    for parts in turns:
        for ev in post("/run", {"app_name": app, "user_id": "harness", "session_id": session["id"],
                                "new_message": {"role": "user", "parts": parts}}) or []:
            usage = ev.get("usageMetadata") or ev.get("usage_metadata") or {}
            if usage:
                rec["model_calls"] += 1
                rec["input_tokens"] += usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0
                rec["output_tokens"] += usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0
            for p in (ev.get("content") or {}).get("parts") or []:
                call = p.get("functionCall") or p.get("function_call")
                resp = p.get("functionResponse") or p.get("function_response")
                if call:
                    rec["tool_calls"].append(call.get("name"))
                if resp and resp.get("name") in ("enroll", "verify"):
                    rec["status"] = (resp.get("response") or {}).get("status")
                if p.get("text") and not p.get("thought"):
                    rec["reply"] = p["text"].strip()
        if rec["status"] is not None:
            break
    return rec


def run_llm_routed(conf: PipelineConfig, *, labels_path: Path, out_path: Path, per_arm: int, seed: int,
                   url: str, out: Callable[[str], None] = print,
                   post: Callable[[str, dict[str, Any]], Any] | None = None) -> Counter:
    """Send `per_arm` harness trials per arm through the orchestrator (Req 10); one record each to `out_path`."""
    labels = [json.loads(x) for x in labels_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    trials = pick_llm_trials(labels, per_arm, seed)
    post = post or _adk_post(url)
    raw = conf.path("raw")
    counts: Counter = Counter()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, label in enumerate(trials):
            rec = scripted_trial(post, label, (raw / label["probe"]).read_bytes())
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts[(label["arm"], rec["status"] or "no_result")] += 1
            if (i + 1) % 10 == 0 or i + 1 == len(trials):
                out(f"{i + 1}/{len(trials)} llm_routed trials")
    return counts


def summary_lines(counts: Counter) -> list[str]:
    return [f"{arm:6} {status:18} {n}" for (arm, status), n in sorted(counts.items())]

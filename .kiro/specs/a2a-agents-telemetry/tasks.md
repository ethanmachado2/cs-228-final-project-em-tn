# Implementation Plan: A2A Agents and Telemetry Capture (Phase 2)

Each task leaves its tests passing and its demo working before the next starts. Host unit tests run with `pytest`. Toolchain tests run with `docker compose run --rm fpv pytest -m "toolchain or dataset or not (toolchain or dataset)"`. Live tests run against `docker compose up -d` from the harness container.

- [x] 1. Dependencies and contract codecs
  - Add `a2a-sdk[http-server]==1.3.0`, `uvicorn`, `httpx` and `scikit-learn` with their transitive closure to the hash-pinned `requirements-pipeline.lock`, resolved with the phase 1 pins as constraints, so that protobuf stays at the TensorFlow pin.
  - Write `src/fpv/agents/__init__.py` and `src/fpv/agents/codec.py`: part encoders and decoders for every contract, plus `ErrorCode` and the exception mapping.
  - Tests: round trip for every contract (arrays byte-identical, floats exact); the error-code mapping in both directions; `pip check` passes and `fpv doctor` is OK.
  - Demo: the container test run passes and `fpv doctor` is green on the rebuilt image.
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Generic stage agent with server-side capture (quality agent first)
  - Write `src/fpv/telemetry.py` (writer, blob store, ASGI middleware, capturing httpx transport) and `src/fpv/agents/serve.py` (`StageExecutor`, AgentCards, app factory, `python -m fpv.agents.serve <name>`).
  - Add the `quality` compose service and the internal `a2a` network.
  - Tests: card shape; valid call `completed` with the same result as `LocalStages`; invalid input `rejected` without running the stage; stage exception `failed` with its code; one JSONL record per request; recorded byte counts equal the wire length; restored bodies equal the originals.
  - Demo: `docker compose up -d quality`. From a container on the a2a network, fetch the card, score one image, and show the hop in `data/telemetry/quality.jsonl` with `jq`.
  - _Requirements: 2.1-2.7, 5.1-5.5_

- [x] 3. Remaining stage agents and `A2AStages`
  - Add the preprocess, extract, adapter and matcher agents and their compose services. Write `src/fpv/agents/client.py` (`A2AClient`, `A2AStages`, the run contextvar).
  - Tests: stage-level parity, `LocalStages` vs `A2AStages`, on 20 seeded Real and Altered samples; remote error codes raise the phase 1 exceptions.
  - Demo: `run_demo` with `A2AStages` against the five live agents prints `MATCH`. Each agent's server record has a matching client record.
  - _Requirements: 2.1, 3.1-3.4, 5.1_

- [x] 4. Controller agent, end-to-end parity and run correlation
  - Add the `controller` agent (`enroll`, `verify`) with uploads by SHA-256, the `input-required` and `rejected` lifecycle, per-`contextId` presentation counts, outcome data parts and `parent_task_id`/`trial_id`/`control_mode` propagation.
  - Tests: end-to-end parity on seeded train trials (in-process vs over A2A); rejection → `input-required` → continuation → cap → `rejected`; every stage hop of a run carries the controller task ID.
  - Demo: enroll a Real image and verify its Altered-Easy variant through the controller from the harness container, get `MATCH`, and show the run's hop trail with `jq`.
  - _Requirements: 4.1-4.7, 5.6, 5.7_

- [x] 5. Always-on compose setup
  - Compose: `a2a` (`internal: true`) and `egress` networks; healthchecks; `restart: unless-stopped`; `depends_on: service_healthy`; startup warm-up flagged `warmup: true`; image digest in every record; a `harness` service under `profiles: [harness]`, the only one that mounts `data/labels`. Gitignore `data/telemetry`, `data/labels` and `data/uploads`. Document the commands and query recipes in `README.md` and `src/fpv/README.md`.
  - Tests: a stage container cannot reach the internet; a restarted agent returns healthy; warm-up records are flagged.
  - Demo: `docker compose up -d`, all services healthy in `docker compose ps`, and a verify works before and after `docker compose restart preprocess`.
  - _Requirements: 5.8, 6.1-6.7_

- [x] 6. Arm trial generator and A2A harness
  - Write `src/fpv/arms.py` and `fpv evaluate --backend a2a --per-arm N` (`local` stays the default).
  - Tests: arm definitions (A claims its own identity; B uses a Real image of a different subject at the same hand and finger); same seed, same list and order; labels never appear in telemetry bodies.
  - Demo: a `--per-arm 20` run completes, and every labeled `trial_id` has its controller hop and stage hops.
  - _Requirements: 7.1-7.6_

- [x] 7. Feature reducer and negative control
  - Write `src/fpv/telemetry_features.py` with `reduce`, `export` and `negative-control` commands (`python -m fpv.telemetry_features ...`).
  - Tests: deterministic rows over fixed logs; tier masks drop exactly the intended columns; warm-up runs excluded.
  - Demo: on the Task 6 data, print AUC per tier for C vs B and the negative-control AUC.
  - _Requirements: 8.1-8.5_

- [x] 8. Orchestrator agent (demo front door)
  - Write `docker/Dockerfile.orchestrator` and `src/fpv_orchestrator/agent.py` (tools, callbacks, model-call telemetry), plus a compose service on the `egress` and `a2a` networks, published on `127.0.0.1:8000`.
  - Tests (fake model): invalid arguments blocked; an unconfirmed claim blocked; a contradicting reply replaced; no image bytes in the model request.
  - Demo: in `adk web`, upload an image, claim user 1 Left index, confirm, and get `Match`. The orchestrator→controller hop and model-call records are in telemetry.
  - _Requirements: 9.1-9.9_

- [x] 9. Scripted `llm_routed` arm
  - `fpv evaluate --backend llm_routed` (`fpv.arms.run_llm_routed`), run from the harness: scripted user turns through the orchestrator's ADK server (`/run`) over a seeded sample of the Task 6 trials, N=25 per arm, with tokens per trial in `data/labels/llm_routed.jsonl`.
  - Tests: a scripted conversation reaches each terminal status; cost is recorded.
  - Demo: the reducer over combined data, comparing `controller` and `llm_routed` telemetry for the same trials.
  - _Requirements: 10.1-10.3_

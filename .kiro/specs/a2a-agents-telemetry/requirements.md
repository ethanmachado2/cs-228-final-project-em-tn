# Requirements Document: A2A Agents and Telemetry Capture (Phase 2)

## Introduction

Phase 2 wraps the deterministic phase 1 `fpv` pipeline (spec `fingerprint-pipeline-baseline`) as long-lived Agent2Agent (A2A) agents and captures every A2A hop, payloads included, for the leakage study in `CS228-project-plan.md`:

> Conditioned on the system's final decision, does the metadata of inter-agent (A2A) communication distinguish a genuine fingerprint presentation from a generatively altered one?

This spec supersedes `multi-agent-trust-telemetry` for implementation. That spec stays as a design reference.

The system has these parts:

- **Stage agents** (`quality`, `preprocess`, `extract`, `adapter`, `matcher`): deterministic A2A servers, one per phase 1 stage, with no LLM.
- **Controller agent**: deterministic. It runs the unchanged phase 1 `PipelineController` over the stage agents and owns all pipeline storage.
- **Harness**: drives labeled experiment trials straight to the controller, with no LLM.
- **Orchestrator agent**: Google ADK with Gemini on Vertex AI. It is the demo front door and the `llm_routed` experiment arm. It only chooses between `enroll` and `verify` on the controller.
- **Telemetry capture**: server-side ASGI middleware in every agent, plus a client-side httpx hook in every A2A client. It writes append-only JSONL and a blob store.
- **Feature reducer**: turns telemetry into one feature row per run, grouped by threat tier.

Out of scope here (a later spec): the trust-model training and analysis (mutual information with a permutation null, conditional-on-decision analysis, dose-response, cross-day replication), the learned generator (arm D), input byte padding, a live auditor view and `orchestrator_vision=on`.

**Security posture.** The system is open and assumed secure. A2A payloads are unobfuscated, and telemetry may contain the raw biometric sample. Agents run on one laptop. Only the orchestrator has outbound network access, and no fingerprint pixels are sent to Vertex AI.

## Glossary

- **A2A**: Agent2Agent protocol, spec 1.0, JSON-RPC binding (`a2a-sdk` 1.3.0).
- **Hop**: one HTTP request/response between an A2A client and an A2A server.
- **Run**: one controller `PipelineController.enroll` or `verify` call, identified by `run_id`.
- **Trial**: one labeled presentation sent by the harness, identified by an opaque `trial_id`.
- **Tiers**: T1 = sizes and timing; T2 = structure and lifecycle; T3 = payload content, including pixels; oracle = match score and decision.
- **Arms**: A = a Real image against its own enrollment (a ceiling); B = a Real image of another subject with the same hand and finger (a zero-effort impostor, the main control); C = altered genuine; C' = altered impostor.
- **`control_mode`**: `controller` (harness → controller) or `llm_routed` (orchestrator → controller).
- **Parity**: identical status, reason, `quality_before`, per-attempt minutiae counts, `configuration_id` and Bozorth3 score between phase 1 in-process stages and phase 2 A2A stages.

## Requirements

### Requirement 1: Contracts over A2A

**User Story:** As a researcher, I want phase 1 stage inputs and outputs carried as typed A2A parts, so that agents exchange exactly what the in-process pipeline exchanges.

#### Acceptance Criteria

1.1 THE SYSTEM SHALL encode `QualityResult`, `AttemptConfig`, `PreprocessResult`, `ExtractionResult`, `XytTemplate` and `MatchResult` as A2A parts, with 2-D uint8 arrays as lossless PNG raw (file) parts and all other fields as data parts.

1.2 WHEN a contract is encoded and decoded THE SYSTEM SHALL return an equal value, with arrays byte-identical and floats exact.

1.3 THE SYSTEM SHALL map phase 1 exceptions to error codes: `QualityToolError` → `nfiq2_tool_error`, `InsufficientMinutiae` → `insufficient_minutiae` (with `which` and `count`), `MatcherError` → `bozorth3_tool_error`, invalid input → `invalid_input`, and any other stage exception → `stage_error` (with its exception type).

1.4 THE SYSTEM SHALL pin `a2a-sdk` and its dependency closure in the hash-pinned `requirements-pipeline.lock` without changing any phase 1 pin.

### Requirement 2: Stage agents

**User Story:** As a researcher, I want each pipeline stage to be an independent A2A server, so that inter-stage communication is observable as A2A traffic.

#### Acceptance Criteria

2.1 THE SYSTEM SHALL serve the quality (skills `assess_quality`, `quality_after`), preprocess, extract, adapter (`adapt`) and matcher (`match`) stages as separate A2A servers, each with an AgentCard at `/.well-known/agent-card.json`.

2.2 THE SYSTEM SHALL execute each skill by calling the unchanged phase 1 function.

2.3 WHEN a skill succeeds THE SYSTEM SHALL end the task `completed` with the encoded result as an artifact.

2.4 WHEN a stage raises THE SYSTEM SHALL end the task `failed` with the error code of Req 1.3.

2.5 WHEN a request does not decode into the skill's input THE SYSTEM SHALL end the task `rejected` with `invalid_input`, without running the stage.

2.6 THE SYSTEM SHALL NOT import or call an LLM in any stage agent or the controller.

2.7 THE SYSTEM SHALL load stage models at startup, not per request.

### Requirement 3: A2A stages client and stage parity

**User Story:** As a researcher, I want the controller to call stages through the same `Stages` protocol, so that phase 1 logic is reused unchanged.

#### Acceptance Criteria

3.1 THE SYSTEM SHALL provide `A2AStages`, an implementation of the phase 1 `Stages` protocol in which each method is one A2A call to the corresponding stage agent.

3.2 WHEN a remote stage fails THE SYSTEM SHALL raise the phase 1 exception for its error code (`QualityToolError`, `InsufficientMinutiae`, `MatcherError`) or a stage error otherwise, so that the phase 1 attempt loop and error mapping behave as in phase 1.

3.3 WHEN the same sample and frozen configuration are processed THE SYSTEM SHALL return stage results from `A2AStages` equal to those from `LocalStages`.

3.4 THE SYSTEM SHALL read peer agent URLs from the environment.

### Requirement 4: Controller agent

**User Story:** As a researcher, I want a deterministic controller agent with `enroll` and `verify` skills, so that both the harness and the orchestrator reach the pipeline through one A2A boundary.

#### Acceptance Criteria

4.1 THE SYSTEM SHALL expose `enroll` and `verify` skills that run the unchanged `PipelineController` with `A2AStages`.

4.2 THE SYSTEM SHALL accept the sample as image bytes in a file part with no filename, plus a data part with the claimed `user_id`, `hand` and `finger`, and SHALL store the bytes as `uploads/<sha256>.bmp`, so that no filename-derived label reaches the pipeline.

4.3 WHEN a run ends `REJECTED_QUALITY` or `REJECTED_MINUTIAE` and the session has presentations left THE SYSTEM SHALL end the task `input-required`, and SHALL treat the next message in the same `contextId` as the next presentation.

4.4 WHEN a session reaches `session.max_presentations` rejected presentations THE SYSTEM SHALL end the task `rejected`.

4.5 WHEN a run ends with any other status THE SYSTEM SHALL end the task `completed` with a data part holding the outcome (status, reason, message, `run_id`, scores and counts).

4.6 THE SYSTEM SHALL be the only writer of its store `data/a2a/` (SQLite, templates, probes) and of `data/uploads`, separate from the phase 1 CLI's `data/db`, so no SQLite file has two writers.

4.7 WHEN the same trials are processed by the in-process controller and by the controller agent THE SYSTEM SHALL produce parity results (`ground_truth_genuine` is expected to be `None` over A2A, because uploads are unlabeled).

### Requirement 5: Telemetry capture

**User Story:** As a researcher, I want every A2A hop captured on both sides with timing, size, structure and content, so that the leakage analysis has the full view an observer could have.

#### Acceptance Criteria

5.1 THE SYSTEM SHALL record every inbound A2A request in every agent (server side) and every outbound A2A request in every A2A client (client side).

5.2 THE SYSTEM SHALL record for each hop: `schema_version`, side, agent and peer, wall-clock start, monotonic start and end in nanoseconds, latency, HTTP status, JSON-RPC method, skill, task ID, context ID, `parent_task_id`, `trial_id`, `control_mode`, final task state, state history, error code, part kinds and counts, request and response HTTP body bytes as sent on the wire, the warm-up flag, the image digest, and the request and response bodies.

5.3 THE SYSTEM SHALL replace each raw part's bytes in a stored body with `{sha256, len}` and SHALL write the bytes once to `data/telemetry/blobs/<sha256>`, so that the original body can be restored exactly.

5.4 THE SYSTEM SHALL write one append-only JSONL file per process role, with exactly one writer per file.

5.5 THE SYSTEM SHALL capture passively: disabling capture SHALL NOT change any pipeline output.

5.6 THE SYSTEM SHALL carry the controller task ID to every stage call of a run as message metadata `parent_task_id`, and SHALL pass the harness's `trial_id` and `control_mode` through to the stage calls.

5.7 THE SYSTEM SHALL NOT place ground-truth labels in any A2A message.

5.8 THE SYSTEM SHALL keep `data/telemetry`, `data/labels` and `data/uploads` out of git.

### Requirement 6: Always-on deployment

**User Story:** As a presenter, I want the agents to stay up between presentations, so that peers can use the system live and models are not reloaded for every sample.

#### Acceptance Criteria

6.1 THE SYSTEM SHALL start every agent with `docker compose up -d` as a long-running service with `restart: unless-stopped` and a healthcheck on its AgentCard.

6.2 THE SYSTEM SHALL run all stage agents and the controller from the phase 1 toolchain image, with one service per agent.

6.3 THE SYSTEM SHALL attach the stage agents, the controller and the harness only to an internal network with no internet access, and SHALL give outbound access only to the orchestrator.

6.4 WHEN a stage agent starts THE SYSTEM SHALL call each of its skills once on itself with a synthetic input, record those hops with `warmup: true`, and report healthy only afterwards. (The controller has no models to load; the harness flags its first trials as warm-up.)

6.5 THE SYSTEM SHALL provide a `harness` service under a compose profile, and SHALL mount `data/labels` only into it.

6.6 THE SYSTEM SHALL keep the phase 1 `fpv` batch service unchanged.

6.7 THE SYSTEM SHALL document start, stop, logs and telemetry query commands.

### Requirement 7: Arms and the A2A harness

**User Story:** As a researcher, I want labeled trials for arms A, B, C and C' sent through the controller, so that genuine-vs-altered leakage can be measured against a zero-effort impostor control.

#### Acceptance Criteria

7.1 THE SYSTEM SHALL generate arm C and C' trials with the phase 1 `generate_trials` (Altered probes, genuine and impostor claims), arm A trials (each enrolled Real image against its own identity) and arm B trials (a Real image of a different subject with the same hand and finger, claiming the enrolled identity).

7.2 WHEN the same seed and split are used THE SYSTEM SHALL generate the same trial list and the same interleaving.

7.3 THE SYSTEM SHALL enroll the chosen subjects' Real images through the controller before any trial, then send trials one at a time in a seeded random interleaving across arms, flagging the first 20 trials as warm-up.

7.4 THE SYSTEM SHALL write one label record per trial (`trial_id`, arm, alteration, difficulty, genuine, subject, claim, split, warm-up) to `data/labels/labels.jsonl`, and only the harness SHALL write it.

7.5 THE SYSTEM SHALL cap trials per arm with `--per-arm N` (default 300) and SHALL disable the probe cache.

7.6 THE SYSTEM SHALL keep `local` as the default `fpv evaluate` backend and add `--backend a2a`.

### Requirement 8: Feature reducer and negative control

**User Story:** As a researcher, I want a reproducible per-run feature table with tier groups, so that the analysis can ablate channels without re-running experiments.

#### Acceptance Criteria

8.1 THE SYSTEM SHALL reduce telemetry JSONL into one feature row per controller run, with a `feature_set_version` and column groups T1, T2, T3 and oracle.

8.2 THE SYSTEM SHALL record the input image byte size and dimensions as covariates, outside the tier groups.

8.3 THE SYSTEM SHALL exclude warm-up records, and SHALL join labels only in the export step.

8.4 WHEN the reducer runs twice over the same logs THE SYSTEM SHALL produce identical rows.

8.5 THE SYSTEM SHALL export the features to CSV, and SHALL run a negative control (arm A randomly split into two labels, L1 logistic regression per tier, cross-validated AUC) next to the C vs B contrast.

### Requirement 9: Orchestrator agent (demo front door)

**User Story:** As a course peer, I want to chat with an agent that collects my sample and claim and returns the pipeline's result, so that I can use the system without the CLI.

#### Acceptance Criteria

9.1 THE SYSTEM SHALL implement the orchestrator as a Google ADK `LlmAgent` on Vertex AI Gemini, with the model ID from `FPV_MODEL_ID` (default `gemini-3.8-flash`) and the project and region from the environment, using Application Default Credentials.

9.2 THE SYSTEM SHALL expose to the model only the tools `confirm_claim`, `enroll` and `verify`, where `enroll` and `verify` call the controller agent over A2A through the same client and capture as the harness.

9.3 THE SYSTEM SHALL validate every tool argument (user ID 1-600, hand in {Left, Right}, finger in {thumb, index, middle, ring, little}, a `sample_ref` that resolves to an upload), and IF validation fails THEN THE SYSTEM SHALL block the call, return an error to the model and log `guardrail_block`.

9.4 IF `enroll` or `verify` is called with a claim that differs from the confirmed claim THEN THE SYSTEM SHALL block the call and log `guardrail_block`.

9.5 IF the model's final reply contradicts the tool's status THEN THE SYSTEM SHALL replace it with the deterministic result message and log `response_substituted`.

9.6 THE SYSTEM SHALL pass uploaded images to the model only as an opaque `sample_ref` and SHALL NOT send pixel data to Vertex AI.

9.7 THE SYSTEM SHALL record every model call (model ID, token counts, latency, tool calls) in the orchestrator telemetry, and SHALL tag its controller calls `control_mode=llm_routed`.

9.8 THE SYSTEM SHALL publish the ADK web UI on `127.0.0.1` only, and SHALL document the change for trusted-LAN access.

9.9 THE SYSTEM SHALL NOT commit credentials or `.env` files.

### Requirement 10: Scripted `llm_routed` arm

**User Story:** As a researcher, I want a small set of trials sent through the orchestrator, so that I can see how an LLM router changes the telemetry.

#### Acceptance Criteria

10.1 THE SYSTEM SHALL drive the orchestrator with scripted user turns for N trials per arm (default 25), sent from the harness through the orchestrator's ADK server, reusing the harness's trial IDs, labels and enrollments.

10.2 THE SYSTEM SHALL record token counts per trial.

10.3 THE SYSTEM SHALL let the reducer process `controller` and `llm_routed` runs together, with `control_mode` as a column.

# Requirements Document: Multi-Agent Trust Telemetry (Phase 2)

> **Status: design reference only. Not scheduled for implementation.** Phase 2 starts after the `fingerprint-pipeline-baseline` spec (phase 1) is complete and its frozen baseline report exists. Criteria here guide decisions and may be refined when phase 2 is scheduled.

## Introduction

Phase 2 turns the phase 1 pipeline into a multi-agent system that follows the Agent2Agent (A2A) protocol, so that we can answer the research question:

> Can A2A interaction leakage act as a signal for trust establishment in a multi-agent fingerprint verification system?

The system has these parts:

- **Orchestrator agent:** Google ADK with a Vertex AI Gemini model. It replaces the CLI as the user interface.
- **Pipeline controller agent:** deterministic.
- **Stage agents:** quality, preprocess, extraction, adapter and matcher. Each wraps one phase 1 module contract and calls no LLM.
- **Telemetry tap and store:** capture A2A interaction telemetry.
- **Trust agent:** acts as a passive bystander/eavesdropper and runs a trust model on telemetry alone.

Two experiment flags vary the system's behavior and are recorded on every run: `control_mode` and `orchestrator_vision`.

The phase 1 CLI remains the batch and evaluation harness.

**Security posture.** The system is open and assumed secure in this phase. A2A messages are not obfuscated, and the trust agent may read message contents. This is intentional, because it is the "leakage" under study.

## Glossary

- **A2A**: Agent2Agent protocol (JSON-RPC over HTTP, AgentCards, tasks, messages, parts).
- **AgentCard**: An agent's published self-description (name, URL, skills, input/output modes).
- **Skill**: A named capability on an AgentCard. Here it maps 1:1 to a phase 1 module function.
- **Hop**: One A2A request/response exchange between two agents.
- **Telemetry tap**: The component that observes A2A traffic and persists telemetry records.
- **Trust agent**: A passive observer that consumes telemetry and outputs a trust score per run.
- **`control_mode`**: `controller` (default) or `llm_driven`.
- **`orchestrator_vision`**: `off` (default) or `on`.
- **Parity**: Identical decisions and scores between phase 2 (controller mode) and phase 1 for the same inputs and frozen config.

## Requirements

### Requirement 1: Stage agents over A2A

**User Story:** As a researcher, I want each pipeline stage exposed as an independent A2A agent, so that inter-stage communication is observable as A2A traffic.

#### Acceptance Criteria

1.1 THE SYSTEM SHALL expose the quality (NFIQ 2), preprocess (pyfing chain), extraction (SBMEX), adapter (`.xyt`) and matcher (Bozorth3) stages as separate A2A server agents.

1.2 THE SYSTEM SHALL publish an AgentCard for each agent whose skills correspond 1:1 to the phase 1 module functions, with input and output schemas derived from the phase 1 frozen dataclasses.

1.3 THE SYSTEM SHALL carry structured inputs and outputs as A2A data parts and images or templates as file parts, and SHALL NOT change the phase 1 stage semantics.

1.4 THE SYSTEM SHALL NOT invoke any LLM inside a stage agent or the controller agent.

1.5 WHEN a stage agent fails THE SYSTEM SHALL return an A2A task in the `failed` state with an error code that maps to the phase 1 error taxonomy.

1.6 WHEN a skill is invoked with an input that does not validate against its schema THE SYSTEM SHALL reject the task without executing the stage.

### Requirement 2: Pipeline controller agent and parity

**User Story:** As a researcher, I want a deterministic controller agent that runs the phase 1 policy over A2A, so that multi-agent results are directly comparable to the baseline.

#### Acceptance Criteria

2.1 THE SYSTEM SHALL provide a controller agent exposing `enroll` and `verify` skills that execute the phase 1 policy (quality gate, bounded retries, adapter, persistence, matching and thresholds) by calling stage agents over A2A.

2.2 THE SYSTEM SHALL keep template storage and run logging in the controller agent, using the phase 1 SQLite schema.

2.3 WHEN the same sample, claim and frozen configuration are processed in `controller` mode THE SYSTEM SHALL produce the same status, `minutiae_count`, `configuration_id` and Bozorth3 score as phase 1.

2.4 THE SYSTEM SHALL propagate a `run_id` and trace context across every hop of a run.

### Requirement 3: Containerization on a single host

**User Story:** As a researcher, I want one container per agent on one machine, so that telemetry reflects agent behavior and not network latency.

#### Acceptance Criteria

3.1 THE SYSTEM SHALL run each agent (orchestrator, controller, five stage agents, telemetry tap and trust agent) in its own container, defined in one compose file on a single host.

3.2 THE SYSTEM SHALL connect agents through one internal container network, and SHALL deny outbound internet access to every container except the orchestrator.

3.3 THE SYSTEM SHALL build stage agent images from a shared base image that contains the phase 1 toolchain, at the pinned phase 1 versions.

3.4 THE SYSTEM SHALL support pinning containers to CPU sets and SHALL record host load alongside each run.

3.5 WHEN the system starts THE SYSTEM SHALL perform documented warm-up calls to each stage agent and SHALL mark warm-up runs so they are excluded from experiments.

### Requirement 4: Orchestrator agent (ADK + Vertex AI Gemini)

**User Story:** As a sandbox user, I want to interact with a conversational agent that collects my sample and identity claim, so that I can enroll or verify without using the CLI.

#### Acceptance Criteria

4.1 THE SYSTEM SHALL implement the orchestrator with Google ADK using a Vertex AI Gemini model, configured by GCP project, region and model ID.

4.2 WHEN a user starts a conversation THE SYSTEM SHALL collect the operation (enroll or verify), a fingerprint image upload, and the claimed user ID, hand and finger.

4.3 THE SYSTEM SHALL return the enrollment or verification result to the user derived verbatim from the controller or policy-wrapper tool output.

4.4 WHEN a sample is rejected THE SYSTEM SHALL request a new sample, up to the phase 1 `max_presentations` limit per session.

4.5 THE SYSTEM SHALL record every orchestrator model call (model ID, token counts, latency, tool calls issued) in the telemetry store.

### Requirement 5: Orchestrator guardrails

**User Story:** As a system owner, I want the LLM constrained so that it cannot bypass the pipeline or invent results.

#### Acceptance Criteria

5.1 THE SYSTEM SHALL validate every tool argument against a schema: user ID an integer from 1 to 600, hand in {Left, Right}, finger in {thumb, index, middle, ring, little}, and a sample reference that resolves to an uploaded file.

5.2 IF a tool call fails schema validation THEN THE SYSTEM SHALL block the call, return a validation error to the model and log a `guardrail_block` event.

5.3 THE SYSTEM SHALL require an explicit user confirmation of the claimed identity before any `enroll` or `verify` call, and IF no confirmation matching the call arguments is recorded in the session THEN THE SYSTEM SHALL block the call.

5.4 THE SYSTEM SHALL NOT allow the model to produce a match, no-match or enrollment decision that differs from the tool result, and IF the final response contradicts the tool status THEN THE SYSTEM SHALL replace it with the deterministic result message.

5.5 THE SYSTEM SHALL accept uploads only as decodable grayscale-convertible images within configured size and dimension limits.

5.6 WHILE `orchestrator_vision` is `off` THE SYSTEM SHALL run a local fingerprint-likeness check (image decodes, dimensions in range, NFIQ 2 able to process it, minimum segmented foreground) before accepting the upload.

### Requirement 6: `control_mode` experiment flag

**User Story:** As a researcher, I want to switch between deterministic and LLM-driven control flow, so that I can measure how LLM control affects telemetry and its trust signal.

#### Acceptance Criteria

6.1 THE SYSTEM SHALL support `control_mode` values `controller` (default) and `llm_driven`.

6.2 WHILE `control_mode` is `controller` THE SYSTEM SHALL expose only the `enroll` and `verify` tools to the model, which delegate to the controller agent.

6.3 WHILE `control_mode` is `llm_driven` THE SYSTEM SHALL expose the individual stage tools to the model through a policy wrapper.

6.4 WHILE `control_mode` is `llm_driven` THE SYSTEM SHALL enforce, in the policy wrapper and not in the model, the stage order (quality gate before preprocessing, preprocessing before extraction, adapter before persist or match), the `max_attempts` bound and the frozen thresholds.

6.5 IF the model issues an out-of-order, over-limit or threshold-overriding call in `llm_driven` mode THEN THE SYSTEM SHALL reject the call, return a policy error to the model and log a `policy_violation` event.

6.6 THE SYSTEM SHALL compute all gate and match decisions in deterministic code in both modes.

### Requirement 7: `orchestrator_vision` experiment flag

**User Story:** As a researcher, I want an optional multimodal path where Gemini sees the fingerprint image, so that I can study its effect on telemetry, with egress made explicit.

#### Acceptance Criteria

7.1 THE SYSTEM SHALL support `orchestrator_vision` values `off` (default) and `on`.

7.2 WHILE `orchestrator_vision` is `off` THE SYSTEM SHALL pass the image to the model only as an opaque sample reference and SHALL NOT send pixel data to Vertex AI.

7.3 WHILE `orchestrator_vision` is `on` THE SYSTEM SHALL send the uploaded image to Gemini as a multimodal input part.

7.4 WHEN `orchestrator_vision` is set to `on` THE SYSTEM SHALL require an explicit acknowledgement setting confirming that biometric data egress to GCP is approved, and IF it is absent THEN THE SYSTEM SHALL refuse to start in that mode.

7.5 THE SYSTEM SHALL keep the deterministic pipeline as the sole source of quality and match decisions regardless of `orchestrator_vision`.

### Requirement 8: Run-level flag recording

**User Story:** As a researcher, I want every run tagged with its experiment conditions, so that telemetry can be stratified and compared.

#### Acceptance Criteria

8.1 THE SYSTEM SHALL record `control_mode`, `orchestrator_vision`, model ID, `config_set_id`, `pipeline_version` and container image digests on every run.

8.2 THE SYSTEM SHALL use the phase 1 `runs.control_mode` and `runs.orchestrator_vision` columns, and SHALL add the remaining fields without breaking the phase 1 schema.

8.3 WHEN the batch harness drives the multi-agent system THE SYSTEM SHALL tag runs with the harness, the seed and the trial ID.

### Requirement 9: Telemetry capture

**User Story:** As a researcher, I want every A2A hop captured with timing, size, structure and content, so that the trust model has a full view of interaction leakage.

#### Acceptance Criteria

9.1 THE SYSTEM SHALL capture every A2A hop between the orchestrator, controller, stage agents and policy wrapper.

9.2 THE SYSTEM SHALL record for each hop: source and destination agent, JSON-RPC method, A2A task ID and context ID, task state transitions, HTTP status, error code, send and receive timestamps (monotonic nanoseconds plus UTC wall time), latency, request and response byte sizes, part kinds and counts, and the payload content.

9.3 THE SYSTEM SHALL capture telemetry passively, so that removing the tap does not change any pipeline output.

9.4 THE SYSTEM SHALL record trace and span identifiers that let hops be grouped into runs and ordered causally.

9.5 THE SYSTEM SHALL measure capture overhead, and IF tap overhead exceeds a documented budget per hop THEN THE SYSTEM SHALL flag affected runs.

### Requirement 10: Telemetry persistence

**User Story:** As a researcher, I want telemetry stored simply and separately from labels, so that the trust model cannot accidentally see ground truth.

#### Acceptance Criteria

10.1 THE SYSTEM SHALL persist telemetry in a SQLite database separate from the pipeline database.

10.2 THE SYSTEM SHALL store ground-truth labels (genuine or impostor, alteration, difficulty) in a separate labels table that is written only by the batch harness.

10.3 THE SYSTEM SHALL NOT grant the trust agent read access to labels at inference time.

10.4 THE SYSTEM SHALL version the telemetry schema and the feature-extraction code.

10.5 THE SYSTEM SHALL support export of telemetry and features to Parquet or CSV for offline analysis.

### Requirement 11: Trust (bystander/eavesdropper) agent

**User Story:** As a researcher, I want a passive agent that watches A2A traffic and estimates trust per run, so that I can test whether interaction leakage carries a trust signal.

#### Acceptance Criteria

11.1 THE SYSTEM SHALL provide a trust agent that reads telemetry only, and SHALL NOT send messages to pipeline agents or influence pipeline decisions.

11.2 WHEN a run completes THE SYSTEM SHALL compute a feature vector for that run from its telemetry and produce a trust score in [0, 1] with a model version.

11.3 THE SYSTEM SHALL expose the trust agent's results through an A2A skill (`assess_trust(run_id)`) and persist them in the telemetry store.

11.4 THE SYSTEM SHALL support offline training of the trust model from exported features and labels.

### Requirement 12: Trust-model experiment

**User Story:** As a researcher, I want a controlled, ablated experiment, so that I can say whether and how much A2A telemetry predicts genuine versus impostor beyond the matcher's own decision.

#### Acceptance Criteria

12.1 THE SYSTEM SHALL use the phase 1 subject-disjoint splits: train the trust model on train-split runs, tune on validation, and evaluate once on test.

12.2 THE SYSTEM SHALL evaluate feature-group ablations: timing only, size only, structure only (hop counts, retries, sequence), visible content only (quality scores, minutiae counts, match score), and all features.

12.3 THE SYSTEM SHALL report ROC AUC, EER, mutual information between features and labels, and calibration for each ablation.

12.4 THE SYSTEM SHALL compare trust-model output with the Bozorth3 decision, reporting agreement and incremental AUC when combined with the match score.

12.5 THE SYSTEM SHALL report results stratified by `control_mode`, `orchestrator_vision`, alteration type and difficulty.

12.6 THE SYSTEM SHALL control documented confounds (LLM nondeterminism, warm-up and caching effects, host load, image dimension effects on payload size, trial ordering) through randomized trial order, repeated runs, warm-up exclusion and stratified reporting.

12.7 WHEN evaluating `llm_driven` mode THE SYSTEM SHALL repeat each trial a configured number of times and report the variance attributable to LLM nondeterminism.

### Requirement 13: Open, unobfuscated system assumption

**User Story:** As a researcher, I want the baseline system deliberately open, so that the measured leakage is an upper bound before any countermeasures.

#### Acceptance Criteria

13.1 THE SYSTEM SHALL transmit A2A payloads without encryption at the application layer, padding or obfuscation in this phase.

13.2 THE SYSTEM SHALL document that the open assumption is intentional and that countermeasures (padding, timing normalization, payload minimization) are future work.

13.3 THE SYSTEM SHALL keep authentication between agents optional and disabled by default in this phase, and SHALL document this as a known risk.

### Requirement 14: Data egress and compliance

**User Story:** As a project owner, I want every path by which biometric data can leave the machine to be explicit and gated.

#### Acceptance Criteria

14.1 THE SYSTEM SHALL document that the orchestrator sends conversation content to Vertex AI in all modes, and fingerprint images only when `orchestrator_vision` is `on`.

14.2 THE SYSTEM SHALL record in the run metadata whether biometric pixel data left the machine.

14.3 THE SYSTEM SHALL require a documented review of the SOCOFing license terms and applicable course or IRB rules before `orchestrator_vision=on` is used.

14.4 THE SYSTEM SHALL read GCP credentials from the environment or Application Default Credentials and SHALL NOT commit credentials to the repository.

### Requirement 15: Batch harness compatibility

**User Story:** As a researcher, I want the phase 1 CLI harness to drive the multi-agent system, so that thousands of labeled trials can be generated cheaply.

#### Acceptance Criteria

15.1 THE SYSTEM SHALL let `fpv evaluate` select a backend (`local` for phase 1, `a2a` for phase 2 through the controller agent, or `orchestrator` for scripted conversations).

15.2 WHEN the `a2a` backend is used THE SYSTEM SHALL generate the same trial list as phase 1 for the same seed and split.

15.3 WHEN the `orchestrator` backend is used THE SYSTEM SHALL drive the orchestrator with scripted user turns and record the cost (tokens) per trial.

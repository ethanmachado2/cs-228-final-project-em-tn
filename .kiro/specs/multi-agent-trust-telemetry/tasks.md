# Implementation Plan: Multi-Agent Trust Telemetry (Phase 2)

> **Future, not scheduled.** These tasks start only after the `fingerprint-pipeline-baseline` spec is complete (frozen config plus `baseline_v1.md`). Task 1 corresponds to Task 13 of the approved plan in `docs/implementation-plan.md`. Before scheduling, re-verify the ADK, a2a-sdk and OpenTelemetry APIs and pin their versions.

- [ ] 1. Architecture document and README (plan Task 13)
  - Write `docs/architecture.md` from this spec's design and the phase 1 design. It should cover:
    - the phase 1 pipeline and its module contracts (as agent skills)
    - the framework comparison and recommendation
    - the agent roster and AgentCards
    - orchestrator guardrails and the two experiment flags
    - the telemetry design and schema
    - the trust agent and the experiment design, with confounds
    - data-handling notes (Vertex egress, SOCOFing license)
    - the container layout
  - Update `README.md` with setup, run and test instructions for phase 1.
  - Tests: none (an optional docs link check).
  - Demo: review with Tyler against the phase 2 scope.
  - _Requirements: 13.2, 13.3, 14.1, 14.3_

- [ ] 2. Shared A2A contracts package
  - Add `fpv_contracts`: pydantic models mirroring the phase 1 frozen dataclasses, PNG and `.xyt` FilePart codecs, JSON schemas for AgentCard skill descriptions, and an error-code enum mapped from the phase 1 error table.
  - Tests: round trip from dataclass to model to JSON and back to the dataclass for every contract. The schemas reject invalid identities and attempt configs.
  - _Requirements: 1.2, 1.3, 1.5, 1.6_

- [ ] 3. Generic stage agent executor and the five stage agents
  - Write `StageAgentExecutor(skill_fn, input_model, output_model)` on a2a-sdk. Expose the quality, preprocess, extract, adapter and matcher agents, each with an AgentCard, startup model loading and warm-up tagging.
  - Tests: contract tests per agent, a failed-task mapping for each phase 1 exception, invalid-input rejection without execution, and confirmation that no LLM dependency is importable in stage images.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.5_

- [ ] 4. Controller agent and parity with phase 1
  - Implement `A2AStages` (the `Stages` protocol over A2A clients) and run the unchanged `PipelineController` behind the `enroll` and `verify` skills. Propagate `run_id` and `traceparent`, and keep storage in the controller.
  - Tests: a parity suite on seeded train-split trials where `LocalStages` and `A2AStages` give identical status, counts, `configuration_id` and score.
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 5. Per-agent containers on a single host
  - Build a shared base image from the phase 1 runtime stage, then per-agent images. Write a compose file with an internal-only network, egress for the orchestrator only, `cpuset` pinning, a host load sampler and recorded image digests.
  - Tests: an egress check (stage containers can't reach the internet), a startup warm-up check, and digests recorded per run.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1_

- [ ] 6. Telemetry tap and telemetry store
  - Build the tap reverse proxy (byte-transparent), a2a-sdk server middleware, an OTel collector exporting to SQLite, telemetry schema migrations, a separate `labels.sqlite` mounted only into the harness, an overhead self-measurement, and Parquet/CSV export.
  - Tests: forwarding is byte-identical, every hop of a run is captured, the overhead budget is checked, removing the tap leaves outputs unchanged, and the trust container can't access labels.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 8.2_

- [ ] 7. Batch harness backends (`a2a`)
  - Add `fpv evaluate --backend a2a`, which uses the same trial generator as phase 1, writes labels to `labels.sqlite`, tags runs with harness, seed and trial ID, and randomizes the interleaving of genuine and impostor trials.
  - Tests: the trial list is identical to phase 1 for the same seed, and labels are written only by the harness.
  - _Requirements: 15.1, 15.2, 8.3, 12.6_

- [ ] 8. Orchestrator agent in `controller` mode with guardrails
  - Build the ADK `LlmAgent` on Vertex Gemini with the `confirm_claim`, `enroll` and `verify` tools. Add schema validation and confirmation checks in `before_tool_callback`, verbatim-result enforcement in `after_tool_callback`/`after_model_callback`, upload checks with a local likeness check, a presentation cap and model-call telemetry.
  - Tests with a fake LLM: invalid args are blocked, unconfirmed calls are blocked, contradicting summaries are substituted, the presentation cap holds, and guardrail events are logged.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 7.2, 14.4_

- [ ] 9. `llm_driven` mode with the policy wrapper
  - Expose the stage tools through a per-run state-machine policy wrapper that applies thresholds and decisions, rejects illegal transitions and logs `policy_violation`.
  - Tests: out-of-order, over-limit and threshold-override attempts are rejected. Legal sequences produce the same outcome as `controller` mode.
  - _Requirements: 6.3, 6.4, 6.5, 6.6_

- [ ] 10. Compliance gate and the `orchestrator_vision` flag
  - Record the SOCOFing license and course/IRB review. Require `FPV_ALLOW_BIOMETRIC_EGRESS=true` to start with vision on, attach the image as a multimodal part when it's on, and record `biometric_egress` per run.
  - Tests: refuses to start without the acknowledgement, sends no pixel data when vision is off (asserted on outgoing model requests), and decisions remain deterministic when it's on.
  - _Requirements: 7.1, 7.3, 7.4, 7.5, 14.1, 14.2, 14.3_

- [ ] 11. Orchestrator harness backend
  - Add `fpv evaluate --backend orchestrator` with scripted user turns, per-trial token cost, and R repeats for `llm_driven` trials.
  - Tests: scripted conversations complete for each status, and cost is recorded.
  - _Requirements: 15.3, 12.7_

- [ ] 12. Trust agent and feature extraction
  - Build the passive trust agent (read-only telemetry, no outbound calls to pipeline agents), versioned feature extraction for the timing, size, structure, content and LLM groups, the `assess_trust(run_id)` skill, and persisted scores.
  - Tests: features are deterministic for fixed telemetry, the agent makes no outbound calls to pipeline agents, and the score range is [0, 1].
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 13. Trust-model experiments and report
  - Train logistic regression and gradient-boosted models on train-split runs, tune on validation, and test once. Run feature-group ablations and report AUC, EER, MI, calibration, incremental AUC over the match score, and agreement with Bozorth3, stratified by `control_mode`, `orchestrator_vision`, alteration and difficulty. Include a confound analysis and LLM variance decomposition.
  - Tests: ablation masks are correct, metric functions are checked against synthetic data, and the test split is used once (guarded).
  - Demo: `docs/trust-experiment-report.md` answering the research question.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 13.1_

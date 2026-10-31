# Implementation status

This is an active implementation, not a completed experiment. The bundle's original README and VALIDATION describe the supplied support scripts, not current implementation acceptance.

Implemented so far: deterministic SQLite creation and state hash; dispatcher and 24 tool routes; strict top-level public schemas; directory/busy queries; versioned calendar/documents/todos; sandbox mail and per-recipient transactional outbox; delayed rule-based approval; stable scheduler and visible event polling; basic independent result grader; public-tool-only basic reference solver; shared budgeted policy loop; durable intent reconciliation.

Verified: 31 CPU unit/integration tests pass. `probe-reference` runs basic and delayed-approval tasks twice and checks matching terminal state hashes. These are software probes, not model evaluations. Latest artifacts: artifacts/reference_probe_v3/.

Still required before completion: full pagination semantics; richer structured user response/event rules; all 14 task templates, dependency DAGs and 1,010-instance split/fingerprint manifests; independent three-layer grading and labelled counterexample suite; official MCP SDK transport; process-isolated world service and real worker-kill fault protocols; simple truncation/full model-based compression and restart context; real model adapters and resource locks; teacher collection, loss masks, SFT training; locked verl adapter and GRPO smoke/update/checkpoint export; RS-SFT/P1; frozen P0/P1 run plans, model evaluation and statistical reports with genuine costs. No training or model performance claims are supported yet.

Commands from this directory:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m officegym.cli probe-reference --out artifacts/NEW_DIRECTORY
```

Probe output directories must be fresh to preserve earlier evidence. Training/API budget fields in configs remain unfilled; null does not authorize unlimited spending.

## Subsequent verified progress

The implementation now includes bounded search/content pagination, structured delayed replies, immutable document publication, meeting completion and delayed minutes, public-tool reference follow-up and rescheduling, reference monitoring boundary handling, explicit document-reference permissions, independent invitation-content/attachment/approval checks, fixed episode call budgets across restarts, isolated WAL-safe rollout clones, labelled grader-case generation/runner, training trajectory/mask gates, bounded effective-group sampling, frozen run-plan primitives, and simple whole-turn context truncation.

Current verification: 61 CPU tests pass. Three reference probe workflows were executed twice with matching state hashes in artifacts/reference_probe_v4. Real SIGKILL worker recovery with surviving world service was verified in artifacts/fault_probe_v2. Eight labelled software grader fixtures were run in artifacts/grader_report_v1 (2 legal, 6 negative, no label mismatch). These limited software checks do not establish full project completion or model performance. Broader template generation, full compression, official MCP integration, locked model/training backends, frozen real experiments and complete audit coverage remain outstanding.

## Dataset implementation checkpoint

Eight train templates T01–T08 and two development combinations D01–D02 now have initial generators. The train templates were generated into 24 probe instances in artifacts/generated_train_templates_v2 and independently replayed in artifacts/train_templates_replay_v2: 24 reference passes, 24 matching final hashes. Development combinations were generated/replayed in artifacts/generated_dev_v1 and artifacts/dev_replay_v1: 6 passes and 6 matching hashes. These are development verification artifacts, not a frozen formal dataset. Early probe manifests used train-range seeds; retain them only as development records. Current generator enforces dedicated seed ranges and rejects partition mixing. Current partitioned probe output: artifacts/generated_probe_partitioned_v1.

Dataset writer preserves required six artifacts, final reference world and file hashes; validate-dataset recomputes semantic fingerprints, checks databases and reference artifacts. reference CLI re-executes from initial worlds. Current regression suite: 73 passing tests. The future-recipient counterexample confirms that globally allowed recipients cannot receive invitations for versions in which they were not participants.

Unfinished scope remains: U01–U04, full operational depth/perturbation design and formal split assembly; complete process/side-effect grading and broad labelled cases; official MCP server; OS worker filesystem isolation; full compression/context reconstruction; real policy/tokenizer adapters; actual SFT/verl training; frozen model evaluations, budgets and reports. Do not infer completion from the increasing unit-test count.

## Model pipeline checkpoint (implementation, not completed experiments)

Official MCP SDK 1.30.0 is installed in .venv and pinned with dependencies in requirements-env.lock. Real SDK stdio tests cover tools/list (24 tools), structured-only observations, trusted operation metadata, replay, and direct/MCP state equivalence. All fourteen initial template generators have passed joint probe generation/reference replay and operation reconciliation (28 instances in artifacts/all_templates_probe_v2 and artifacts/all_templates_replay_v2).

Harness additions: structured summary component with archive/checksums and usage accounting; same-policy enforcement; complete-turn truncation; append-log restoration with task/model/budget checks; persisted parse retries; action-boundary logging and uncertain-write reconciliation. Full Harness real SIGKILL software probe: artifacts/harness_fault_probe_v2 (surviving world service, one outbox effect, independent grade pass). This uses scripted policies, not a measured model baseline.

Pipeline additions: configured local inference adapter, exact prompt-tokenizer interface, single-episode evaluator, hashed-plan batch evaluator with reservations/resume, result export guards, bounded teacher collection with retained infrastructure failures, teacher extraction, native assistant-mask alignment, build-sft manifests/rejections, locked TRL SFT entrypoint, checkpoint hashing/branch checks, bounded GRPO group sampler, and reward audit ledger. Current CPU/protocol regression suite: 107 tests passing. TRL optimizer code has not been exercised; tokenizer tests use protocol fixtures, not actual Qwen tokens.

Outstanding required work includes: formal balanced depth/perturbation generation and combo holdout/family audit; broader adversarial/alternative-solution coverage; OS-level worker isolation; actual model/tokenizer downloads and dependency compatibility probes; genuine teacher traces and manual token/mask review; SFT optimizer smoke/overfit/export equivalence/dev; concrete locked verl adapter and rollout/update smoke; GRPO/RS-SFT training; frozen P0/P1 runs and genuine statistical/curve/cost reports. Model service, API identity and download/training/API budgets were requested from the user and are not yet supplied. CPU implementation work can continue independently.

## Current data and backend checkpoint

artifacts/dataset_development_v1 contains 1,010 reference-verified development instances (800 train, 80 dev, 60 ID, 30 combo, 40 unseen). Independent replay in artifacts/dataset_reference_replay_v1 passed 1,010 grades, 1,010 terminal hashes and 1,010 operation audits. Composition report: artifacts/dataset_development_composition_v1.json (192 scenario families, no cross-split scenario/fingerprint families). This dataset is explicitly NOT frozen: ID scan uses T04 only and combo condition coverage is limited. Five-cell controlled generation and three authored skip-stage probes exist, but they do not prove exhaustive dependency coverage.

Training additions include strict sample revalidation, checkpoint lineage, bounded group sampling, isolated rollout environments with raw-token/logprob/mask and world archives, worker-local registry, and a source-inspected BaseTool/AgentLoop wrapper against verl commit 8df5baeed0f5ecdd7f870d83bc3d79194816e2a9. Upstream source inspection revealed per-call release; the wrapper now holds an episode lease. These wrappers are not installed/runtime-smoke-verified. GRPO launcher checks source/dependency/config identity and finite requested budgets, terminates process groups on timeout, and requires semantic smoke evidence for training. Runtime global rollout-token enforcement, dynamic-group integration and final Hydra mapping remain unverified.

Current regression suite: 116 passing tests. No model download, real tokenizer test, teacher API sampling, GPU optimizer update or frozen model evaluation has occurred. User resource/API budget clarification remains pending; CPU/backend implementation can continue. Remaining broad gates include full adversarial coverage, formal data freeze, OS worker isolation, actual backend compatibility, SFT training/export equivalence, real GRPO smoke/train/RS-SFT and full P0/P1 reports. Source-level adapters and passing CPU tests are not proof these gates are achieved.

## Generation-budget and grader coverage checkpoint

Added a transactional SQLite rollout-token reservation ledger and per-generation/context caps to the source-inspected verl loop. Reservations remain charged after uncertain failures; measured native output-token counts are recorded when known. CPU tests cover frozen limits, cross-connection reservations and crash accounting. Actual verl/vLLM execution and enforcement of the backend max_tokens override remain unverified.

Expanded the labelled grader suite to 12 tool-executed cases: two legal alternatives and ten negatives, including stale materials, missing attachments, an eight-hour scheduling offset and cancellation of an unrelated pre-existing meeting. The last case supplies an initial snapshot for independent mutation comparison. artifacts/grader_cases_v3 and artifacts/grader_report_v3 preserve action traces and grades: 10/10 negatives detected, zero legal false positives, zero label mismatches. These are authored software fixtures, not model-discovered exploits; the interrupted v2 generation is incomplete and is not a report source.

Full CPU/protocol regression suite: 118 tests passed. No real model/training or frozen experimental gate is newly satisfied by this checkpoint.

## Trusted rollout resource contract checkpoint

The GRPO launcher now requires and validates rollout_contract: global limit must equal the launch max_rollout_tokens, per-trajectory reservations must fit, initial actor must match the starting checkpoint, and ledger/archive paths must be absolute. training/rollout_contract.py provides copy-on-bind row metadata with conflict rejection so data rows cannot silently replace these limits. Regression coverage includes budget/actor mismatches, relative paths and per-row overrides; full suite: 120 tests passed.

This is preflight and binding infrastructure, not proof of end-to-end budget enforcement: the concrete upstream dataset serialization/loading entrypoint must still invoke the binding function, updated actor versions must be supplied by real synchronization, and multi-host storage semantics and vLLM generation caps must be runtime-tested. No GPU or model experimental gate is complete.

## Concrete verl input preparation checkpoint

training.prepare_rl now validates the dataset manifest, selects only train entries, builds public full-harness prompts and binds the shared resource contract into each officegym row. It writes JSONL plus source/data hashes and a preparation report. The GRPO launcher checks the prepared file hash/contract/split and requires the resolved upstream data.train_files to point to that exact input. Negative checks reject changed data and alternate upstream paths.

Source inspection of pinned verl RLHFDataset confirms JSONL loading and preservation of officegym row metadata; AgentLoop workers forward non-tensor row fields. artifacts/rl_input_probe/prepared contains 800 prepared train rows from development_dataset_v1, independently checked against each source task's train split. The actor marker is software_preparation_probe_no_model, not a checkpoint or training result. Runtime backend loading, actor synchronization, effective-group update integration and real training remain unverified. Full CPU/protocol suite: 120 tests passed.

## Training tool protocol parity checkpoint

Pinned upstream ToolAgentLoop gathers multiple calls concurrently and returns parse/unknown-tool errors without invoking the world. OfficeGymAgentLoop now rejects multiple tool calls before execution, caps their terminal reward at failure, and overrides single-call execution to pass unknown names and malformed JSON through the canonical dispatcher. Structured observations are preserved without upstream character truncation. Episode leases remain responsible for world cleanup, including when ledger finalization fails.

A real SQLite dispatcher regression verifies malformed JSON and unknown names consume two calls and sixty simulated seconds without meeting effects. Full CPU/protocol suite: 121 tests passed. This verifies the shared execution helper, not imported/running verl: the optional GPU dependency stack and complete adapter lifecycle still require runtime smoke validation. No real model or training gate is satisfied here.

## Material-update atomic coverage correction

Requirement audit found that the old training T05 only read static material, while the combo holdout required DocumentVersionPublished. T05 now includes a single material-publication event and a monitoring horizon; the reference must refresh the meeting to version 2. Training still excludes the ordered AttendeeChanged → DocumentVersionPublished signature. Dataset validation now rejects combined train/combo manifests containing combo event kinds absent from training; composition reports expose the exact atom coverage. Tests remove T05 to verify this failure is detected.

artifacts/dataset_development_v2 replaces v1 as the current development candidate: 1,010 instances, all reference grades passed, all 1,010 fresh replay hashes matched and all 1,010 operation audits passed (artifacts/dataset_reference_replay_v2/summary.json). Composition v2 confirms both combo atoms occur separately in training. v1 and its prepared RL inputs are retained as historical artifacts and must not be used as a freeze candidate because of the missing training atom.

The combo holdout itself requires two perturbation events; assigning it to P0/P1 cells would mislabel its condition. Its current single-template composition and restricted interpretation are explicit in the manifest; five-cell combo stratification is optional in the handbook and incompatible conditions are not forced. ID still uses the T04 five-cell composition. Dataset remains unfrozen pending broader dependency/noise audits. Full CPU/protocol suite: 122 tests passed. No model training/evaluation claim follows from these reference results.

## Dependency graph and ordered holdout audit

Generator DAGs previously serialized all stages into one chain, incorrectly asserting an order between material reading and reply collection (and contradicting the reference solver order). The graph now makes these independent prerequisites of planning, retains confirmation/invitation and followup dependencies, and binds handoff evidence to confirmation. This corrects claimed path lengths; it does not prove every information prerequisite is unavoidable for all policies. Existing v2 dataset artifacts preserve the earlier DAG representation and need regeneration before freezing.

Ordered holdout checking previously used whole-list equality, allowing extra events to hide a forbidden combination. It now rejects the forbidden ordered subsequence even with intervening or surrounding events; reverse order remains a different signature. Tests cover this leakage path explicitly. artifacts/dependency_dag_probe_v1 and dependency_dag_replay_v1 verify all fourteen templates, all terminal hashes and operation audits after the correction. Full CPU/protocol suite: 125 tests passed. Broader stage-skip evidence, actual model runs and final data freeze remain incomplete.

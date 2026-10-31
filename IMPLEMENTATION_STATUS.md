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

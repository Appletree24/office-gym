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

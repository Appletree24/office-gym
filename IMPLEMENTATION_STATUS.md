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

## Installed distribution checkpoint

Found and fixed source-tree-only runtime dependencies: world DDL, runtime journal DDL and tool contracts now use importlib.resources and ship inside the wheel. Repository schema/contracts paths are symlinks to the canonical package resources. Added the officegym console entrypoint. Installed provenance hashes package Python/SQL/JSON files rather than an absent repository src directory, and reports unknown git state as null for distributions.

Built artifacts/wheel_probe_v1/officegym-0.1.0-py3-none-any.whl (SHA256 a0f081aac9961515ecccf5b8f0c22c4f1843af9b95ce21a01788a6c4fa04e043). Installed without dependencies in a new /tmp virtualenv, unset PYTHONPATH, and ran from /tmp: all basic/approval/followup reference probes passed; journal initialization/write-intent preparation, all 24 packaged contracts and installed provenance passed. Evidence is in artifacts/wheel_probe_v1/installed_reference and installed_provenance.json. Source regression suite: 125 tests passed. This verifies core wheel installation; optional MCP/training GPU stacks and full experimental gates remain separate and incomplete.

## Availability evidence scope correction

Independent grading formerly accepted any reply from each participant, even if it was invisible to the principal, unrelated to an availability request, or explicitly incomplete. Availability evidence now requires a reply addressed to the principal, a provided response with a valid containing window, and a preceding availability request in the same thread sent by the principal to that participant. Malformed unrelated correspondence is ignored rather than crashing grading. Tests distinguish unrequested, invisible, incomplete, malformed and legitimate requested replies.

Scoring identities are now shared constants: eval-v2-reply-evidence and reward-v2-reply-evidence. The numeric terminal mapping remains 1/0/-1; corrected evidence acceptance changes scoring semantics, so older grades/training rewards must not silently mix with this revision. All fourteen reference instances were re-executed under the new grader in artifacts/reply_evidence_replay_v1: grades, deterministic terminal hashes and operation audits all passed. Full regression suite: 126 tests passed. Full development artifacts still require regeneration/regrading before freeze; no actual trained checkpoints exist to reevaluate.

## Scoring compatibility and consolidated development v3

Current scoring identities are now required by dataset validation, teacher extraction, SFT sample construction and SFT optimizer preflight. SFT dataset manifests record those identities. Missing/old versions are rejected even when a reference-grade file is rehashed or a dataset has a passed manual-review marker; reuse requires regrading original worlds rather than relabeling old success. Regression suite: 129 tests passed.

artifacts/dataset_development_v3 is the current unfrozen candidate, regenerated with corrected dependency DAGs, material-event coverage and eval-v2/reward-v2 reply evidence. It contains exactly 800/80/60/30/40 instances. artifacts/dataset_reference_replay_v3/summary.json confirms 1,010 reference successes, 1,010 terminal-hash matches and 1,010 consistent operation audits. Composition v3 confirms no missing combo atoms in training. artifacts/rl_input_probe_v3 contains 800 fresh contract-bound train rows (software actor marker, not real checkpoint).

artifacts/dependency_skip_probe_v3 covers all 12 D3 ID instances: 36 authored attempts skipping confirmation, reply collection or followup todos, all rejected. This extends the earlier single-instance coverage but does not prove material-information or every possible dependency unavoidable, and is not a model-discovered exploit report. Older v1/v2 artifacts and RL preparation remain historical. Formal freeze, remaining environment/isolation audits, actual tokenizer/backend compatibility, teacher generation, SFT/GRPO/RS-SFT and full measured P0/P1 reports remain unfinished.

## Export/statistics comparability correction

Measured-result schema/export now include required model_version, reward_version and harness_version. Statistical selection rejects changing model versions within a group and rejects simple/full comparisons of the same model family using different model versions; distinct model/checkpoint versions remain permitted across training groups. Reports preserve each group's model identity. Global version consistency also includes reward and Harness identities.

Export budget identity now hashes the recorded episode budget plus max_context_tokens and max_output_tokens from resolved_eval_config.json, and rejects disagreement between recorded and configured episode limits. Previously contexts/output caps could differ while budget_hash stayed equal. Grade identities must also match recorded provenance. Updated schema and fixture tests cover version drift and these budget mismatches. Full regression suite: 133 tests passed. These are fabricated unit-test rows, not model measurements or a formal statistical report. Actual model evaluations and associated cost/curve reports remain pending.

## Episode crash-window correction

Configured evaluation previously wrote resolved_eval_config/tokenizer_manifest only after run_episode had published result.json. A crash in between could leave a terminal result that batch resume could not export. Those metadata artifacts are now handed into run_episode and durably written before any model call, alongside provenance and budgets. Metadata names are restricted; unsupported/non-JSON values are rejected before episode output creation.

Added atomic JSON publication (fsync temporary file, replace, fsync directory) for setup records and terminal result. Failure-injection tests verify a publish error retains the previous complete JSON and removes its temporary file; the episode integration test verifies metadata exists when policy.generate is first called. Full regression suite: 134 tests passed. No inference-service or real experiment result was produced by these software fixtures. Formal GPU/model/experiment gates remain incomplete.

## Malformed model-envelope accounting correction

ChatPolicy previously indexed choices/message directly, so malformed response envelopes could raise after measured usage was available and lose that usage from the harness ledger. It now treats malformed choices, messages, tool-call and function envelopes as unparseable turns while preserving raw responses and measured input/output counts. Summary parsing also preserves measured usage and raw response on malformed content/envelopes; FullContext includes that raw response in context-selection evidence even when an invalid summary falls back.

A transport-fixture integration test exercises ChatPolicy through Harness: three malformed responses terminate at the configured parse-retry bound, account for 27 input/21 output tokens, preserve all raw responses, and execute no tools. Envelope-shape tests cover missing/empty choices, null messages/calls/functions and malformed summary responses. Full CPU/protocol regression suite: 136 tests passed. These fixture token counts are accounting checks, not model measurements. Actual model/API collection and training remain unfinished.

## Assistant-text and teacher source-fidelity correction

ChatPolicy/Harness previously discarded assistant content accompanying a tool call. Such content now survives response logging, subsequent request rendering, interaction records and uncertain-write recovery. Intent action records retain it before side effects. Harness identity is now harness-v2-assistant-content; restore and teacher extraction reject missing/older identities. Provenance and SFT source/dataset manifests carry the version, and optimizer preflight rejects older source Harness manifests.

Teacher extraction now matches each original model response to the action and its interaction tool name/arguments/content, requires raw model-response evidence and rejects unmatched responses/duplicate pending identities. Parse-retried episodes are excluded because their discarded malformed assistant output cannot be losslessly reconstructed into the SFT conversation. Tests use explicit authored protocol fixtures to verify content preservation and reject changed interaction arguments/retried traces. Recovery tests check content for both committed and precommit response loss.

Full regression suite: 138 tests passed. artifacts/harness_fault_content_v1 reruns the actual SIGKILL software probe under the new Harness/scoring identities: worker exit -9, surviving world, one outbox effect, RECONCILED journal and independent success grade. This remains scripted recovery evidence, not model performance. Real tokenizer alignment, teacher sampling, optimizer/backend smoke and formal experiments remain unfinished.

## GRPO execution/config binding correction

Launch preflight formerly checked an upstream config hash but accepted arbitrary argv that could ignore it. Commands must now equal bound_command(): the current interpreter running officegym.training.verl_entry with the absolute resolved config path and its hash. The entrypoint rechecks bytes immediately before loading, rejects defaults/unresolved interpolation, and invokes the pinned upstream Hydra-decorated main's unwrapped function so its validation/task-runner selection execute without new CLI overrides. The exact upstream main_ppo.py source was inspected at the pinned commit; this path is not runtime-backend verified.

Also fixed an impossible smoke gate: exact config hash equality prevented a <=5-update smoke from authorizing a longer training run. compatibility_hash now permits only trainer.total_training_steps to differ; all other resolved settings remain bound. Semantic smoke reports must attest that compatibility hash; launch archives the resolved config and both identities. Tests verify arbitrary commands, changed bytes and unresolved interpolation are rejected, 5→100 updates can pass the compatibility gate with authored smoke fixtures, and learning-rate changes alter compatibility. Full regression suite: 140 tests passed. Authored smoke-report markers are test fixtures, not actual rollout/update/export evidence; no real GRPO launch occurred.

## Worker registration and schema fidelity checkpoint

Pinned AgentLoop workers load custom loops from rollout.agent.agent_loop_config_path and tools from multi_turn.tool_config_path; driver-local registration is insufficient. prepare_rl now writes hashed agent_loop.json and tools.json with the qualified OfficeGym target and exactly 24 native BaseTool entries. Launch validation requires resolved paths to match those files, checks contents against current tool contracts, and rejects additional function tools. Tests cover alternate paths, rehashed missing tools and extra function tools. Full regression suite: 141 tests passed.

artifacts/rl_worker_registration_v1 contains 800 fresh contract-bound train rows and both registration files. Standalone execution of the pinned upstream Pydantic schemas module accepted all 24 configurations but roundtripped all 24 schemas lossily (including additionalProperties and array items). OfficeGymAgentLoop now overrides prompt tool_schemas with canonical complete OpenAI-format contracts and rejects per-sample tool_selection, retaining consistent tool exposure. schema_probe.json records source identity and observed lossy conversions. This is source/isolated-schema evidence, not a full verl import or Ray/vLLM rollout. Worker filesystem availability, actual backend loading, training updates and remaining experimental gates are still unverified.

## Native rollout terminal consistency correction

The verl adapter formerly clamped truncated/budget/invalid-batch rewards only after RolloutEnvironment had archived a possibly successful terminal.json. Termination reasons now enter environment.finalize, which preserves domain_grade separately and emits one consistent effective grade/reward for both archives and optimizer output. Terminal records also include independent operation reconciliation; inconsistent operation evidence cannot count as success.

Final-world backup now precedes atomic terminal publication. If terminal publication fails, close retains the world with an incomplete marker, and cleanup closes/removes temporary resources even if archival itself raises. SQLite/reference integration tests cover a domain-correct episode under normal, response-limit, generation-budget and invalid-batch terminations, checking exact returned/archive equality, plus simulated terminal-publication failure. Full regression suite: 143 tests passed. These are software/token fixtures; real native-token backend execution and GRPO optimizer integration remain unverified.

## Prepared RL row consumption gate

Launch preflight previously accepted a matching train-file hash without checking its rows (the old unit launch fixture even used an empty object row). validate_preparation now checks the source manifest hash and complete dataset validation, rebuilds the expected train-only rows, and compares the actual ordered JSONL against task paths/public prompts/resources/metadata exactly. Row count, preparation format, Harness and scoring versions are mandatory; old preparation reports require regeneration. Launch tests now use genuinely generated task artifacts and prepare(), rather than arbitrary JSON placeholders.

Tests explicitly rehash mutated data with a higher tool budget, replaced prompt, alternate task path or removed rows; all are rejected before launching. Full regression suite: 144 tests passed. artifacts/rl_verified_rows_v1 regenerated and revalidated all 800 development train rows, worker plugins and current version identities. The software actor marker remains not a checkpoint; actual worker storage availability, backend rollout/update, teacher sampling/training and formal experiments remain unfinished.

## Consumer-side partition and seed audit

Template partitions, seed ranges and heldout signatures are now shared task contracts. Dataset validation and direct RolloutEnvironment construction independently enforce the template/split and formal seed-range rules; relabeling U01 as train or reusing a probe seed in a formal split no longer passes merely because task and manifest agree. Probe-only authored tasks may omit a seed, preserving their explicit fixture status. A manifest cannot empty/change the fixed ordered holdout set to disable training exclusion.

Tests rehash relabeled task artifacts to verify both the manifest validator and direct rollout boundary reject unseen-template leakage; they also verify cross-split seed reuse and removed holdout rules are rejected. Full regression suite: 146 tests passed. artifacts/partition_audit_v1.json revalidates all 1,010 current development instances under these checks (800/80/60/30/40). This checks metadata/contracts and artifacts, not all semantic dependency or experimental freeze gates. Actual model/backend experiments and remaining implementation gates are still incomplete.

## Event-label/world consistency checkpoint

Dataset validation now independently compares declared scheduled events with initial SQLite event rows, including due time, priority, identity, payload, preconditions, visibility and pending status. It checks the task event signature and perturbation count against that schedule, verifies declared rule metadata and checks the initial public task against task.json. Ordered holdout checking uses due time/priority/id order. This closes a gap where hashes/fingerprints could be internally consistent while task labels or declared rules did not describe the executed world.

Tests rehash a changed perturbation label and separately alter event priority, approval-rule metadata and initial public task; all inconsistent artifacts are rejected. Full regression suite: 148 tests passed. All 1,010 development v3 instances pass the stronger validator; evidence is artifacts/event_contract_audit_v1.json. This proves schedule/config/label consistency, not that every policy experiences the same number of perturbations or that all depth dependencies are unavoidable. Formal data freeze and actual backend/model/training/evaluation gates remain incomplete.

## Explicit RS-SFT branch implementation

Added training.rs_sft collect/build commands using the existing finite collector, current grader/teacher gates and real-tokenizer SFT builder. Sampling requires full Harness and a model_version equal to the content hash of an unchanged exported stage=sft checkpoint; branch/source/config hashes are retained before collection. Build rechecks the source export/checkpoint, confines successful run paths to the collection and rejects other sampling-model identities. Dataset manifests preserve source lineage.

SFT optimizer preflight now requires matching rs_sft plan/dataset tags and the same original exported SFT weights, rather than a GRPO or replacement checkpoint. Training artifacts preserve the dataset manifest and branch identity. Tests use explicitly non-model byte fixtures and a mocked collector to verify immutable origin, GRPO-stage rejection, finite-cap forwarding and optimizer branch/source checks. Full regression suite: 151 tests passed. No actual rejection sampling, tokenizer run, optimizer step or RS-SFT result has occurred; real checkpoint equivalence and sampling-service identity still need runtime evidence and budgeted experiments.

## Collection resource and reservation ledger completion

The shared teacher/RS-SFT collector now fsyncs each full-token reservation before invoking inference, freezes its sampling config into the collection directory and records source/config hashes. It records generated/input tokens, tool calls, reported wall time and collector-measured elapsed time for every attempt, including failures. Resource summaries carry unknown-attempt counts; reported sums are explicitly lower bounds when usage is missing. GPU-hours/API cost remain null when unmeasured. A per-episode generation-cap violation stops further collection and cannot enter successful_runs.

Tests verify reservation visibility before the callback, fixed config use despite original-config mutation, aggregation across both failed and successful attempts, preserved unknown costs after outages, and immediate stop on a backend overrun. Full regression suite: 153 tests passed. Resource numbers in those tests are authored fixtures, not actual sampling costs; real model collection, GPU usage and monetary accounting still need live experiments.

## Dev-only checkpoint selection implementation

Added eval.selection freeze/select commands. Freeze binds unique stage-appropriate checkpoint contents, export-record hashes, declared optimizer steps and a hashed dev-only three-repeat plan. Selection requires all candidates' complete planned results, checks model identities and common environment/grader/reward/Harness/dataset/budget versions, includes failures in the instance macro mean, and deterministically chooses highest dev success then earliest step/hash. It retains all candidate scores/results hashes, rejects test splits, missing runs and changed candidates, and writes selection once. It cannot independently prove freeze timestamps or authenticity of a model run; retain original run artifacts/command logs.

Moved the existing statistics implementation into installable officegym.eval.analysis, retaining scripts/analyze_results.py as a compatibility launcher. Its result schema is now a canonical packaged resource with the repository contract path linked to it. Existing statistics tests continue through the compatibility entrypoint. An isolated wheel installation from artifacts/wheel_analysis_v1 loaded the analysis CLI and packaged result schema from /tmp with PYTHONPATH unset. This checks packaging/imports, not measured reports.

Full regression suite: 157 tests passed, including authored selection fixtures for tie handling, missing runs, test leakage, budget incompatibility and checkpoint changes. No real checkpoint was selected, no dev/model report was generated, and actual training/frozen evaluation remain unfinished.

## SFT checkpoint retention and curve evidence implementation

SFT formerly used save_strategy=no and only wrote a final adapter. Resolved plans now require checkpoint_interval_steps within the finite optimizer budget; trainer configuration saves at that interval with no automatic checkpoint eviction. Callback integration appends/fsyncs every logged training event, records full saved-checkpoint file hashes/optimizer steps, preserves trainer log history even when train() raises, and inventories the final adapter. Training failures record their exception type. Nonfinite metrics are preserved as explicit markers instead of silently filtered or invalidating JSON publication.

Official Trainer/callback documentation was checked for step-based saving, retention and callback/state semantics: https://huggingface.co/docs/transformers/main_classes/trainer and https://huggingface.co/docs/transformers/main_classes/callback . This is documentation-level compatibility; the operator-locked TRL/Transformers runtime remains untested. Tests use byte/log fixtures, validating retained logs/checkpoint changes/nonfinite markers and rejecting missing/out-of-budget save intervals. Full regression suite: 159 tests passed. No optimizer, real saved checkpoint, merged export or measured training curve was produced by this implementation turn.

## Checkpoint step/export source linkage

Candidate optimizer steps formerly came solely from the selection input. Non-base record_export now requires a saved training event matching the actual source-checkpoint inventory and optimizer step; equivalence-report metadata must bind both that source and the exported model content. Export records preserve a prefix hash/byte length of the training log, allowing later append-only events without invalidating the earlier saved checkpoint. Selection rule format v2 verifies source evidence at freeze and again at selection, rejecting hand-altered candidate steps/history/weights.

Native inventory supports adapter/HF weights and the model_world_size_*_rank_*.pt shard convention inspected in pinned verl fsdp_checkpoint_manager.py. This recognizes file evidence, not valid tensors or a working FSDP runtime. Tests use explicit byte/report fixtures, covering appended logs, changed history, export-weight mismatch and candidate-step mismatch. Full regression suite: 162 tests passed. Real export equivalence generation and GRPO checkpoint event integration still require actual backend execution; no real model export or checkpoint selection occurred.

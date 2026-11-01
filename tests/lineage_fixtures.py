"""Authored byte/config fixtures for lineage tests; no model/teacher/training proof."""
import json
from pathlib import Path
from officegym.training.admission import admit_training,digest
from officegym.training.evidence import TrainingEvidence
from officegym.training.checkpoints import checkpoint_fingerprint,record_export
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
from officegym.harness.versions import HARNESS_VERSION


def admit_fixture(root,evidence):
    root=Path(root);base=root/'fixture_base';base.mkdir()
    (base/'config.json').write_text('{}');(base/'model.safetensors').write_bytes(b'authored fixture base, not model weights')
    data=root/'fixture_data';data.mkdir();(data/'samples.jsonl').write_text('authored format fixture, not teacher data\n')
    manifest=dict(kind='sft_dataset_not_training_result',accepted=[{'fixture':True}],manual_alignment_review_status='passed',samples_sha256=digest(data/'samples.jsonl'),harness_version=HARNESS_VERSION,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION)
    from review_fixtures import reviewed_fixture
    reviewed_fixture(data,manifest)
    config=dict(model_path=str(base.resolve()),model_revision='authored-fixture',max_steps=100,max_wall_seconds=60,locked_packages={k:'fixture-not-installed' for k in ('torch','transformers','trl','peft','datasets')},max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=.0001,lora={},checkpoint_interval_steps=1)
    path=root/'resolved_train_config.json';path.write_text(json.dumps(config))
    return admit_training(evidence,stage='sft',data_manifest=data/'manifest.json',config=path)


def exported_sft_fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);evidence=TrainingEvidence(root);admission=admit_fixture(root,evidence)
    checkpoint=root/'checkpoint';checkpoint.mkdir();(checkpoint/'config.json').write_text('{}');(checkpoint/'model.safetensors').write_bytes(b'authored fixture checkpoint, not trained model')
    source=evidence.checkpoint(checkpoint,1)
    report=root/'equivalence.json';report.write_text(json.dumps(dict(passed=True,checks=['authored-unit-fixture-only'],exported_checkpoint_sha256=checkpoint_fingerprint(checkpoint)['checkpoint_sha256'],source_checkpoint_files_sha256=source['checkpoint_files_sha256'],optimizer_step=1)))
    path=root/'export.json';record=record_export(checkpoint,path,stage='sft',parent_hash=admission['parent_checkpoint_sha256'],training_manifest_hash=admission['data_manifest']['sha256'],versions={},equivalence_report=report,training_events=root/'training_events.jsonl',source_checkpoint=checkpoint,optimizer_step=1)
    return checkpoint,path,record


def checkpoint_profile(root,maximum=5):
    root=Path(root).resolve()
    return dict(data={'train_batch_size':2},trainer=dict(total_training_steps=maximum,use_v1=False,save_freq=1,officegym_evidence_root=str(root),default_local_dir=str(root/'checkpoints'),officegym_verl_checkout=str(root/'verl'),resume_mode='disable'),actor_rollout_ref=dict(rollout={'n':2},actor=dict(strategy='fsdp',ppo_epochs=1,ppo_mini_batch_size=2,checkpoint=dict(async_save=False,save_contents=['model','optimizer','extra']))))

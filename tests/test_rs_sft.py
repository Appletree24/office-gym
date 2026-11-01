"""Branch guards use authored byte fixtures, never real model/optimizer evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from officegym.training.checkpoints import checkpoint_fingerprint
from officegym.training.rs_sft import require_sft_source,collect_rs


class RSSFTTests(unittest.TestCase):
    def source(self,root):
        from lineage_fixtures import exported_sft_fixture
        return exported_sft_fixture(root/'admitted_source')

    def test_grpo_stage_and_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint,export,record=self.source(root)
            self.assertEqual(require_sft_source(checkpoint,export)['checkpoint_sha256'],record['checkpoint_sha256'])
            export.write_text(json.dumps({**record,'stage':'grpo'}))
            with self.assertRaisesRegex(ValueError,'unchanged exported SFT'):
                require_sft_source(checkpoint,export)
            export.write_text(json.dumps(record))
            (checkpoint/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'unchanged exported SFT'):
                require_sft_source(checkpoint,export)

    def test_collection_binds_origin_and_forwards_all_finite_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint,export,record=self.source(root)
            config=root/'config.json'
            config.write_text(json.dumps(dict(mode='full',model_version=record['checkpoint_sha256'])))
            with patch('officegym.training.rs_sft.collect',return_value={'attempts':0}) as collector:
                collect_rs('manifest',config,checkpoint,export,root/'out',max_attempts_per_task=3,max_episodes=20,max_generated_tokens=1000)
                self.assertEqual(collector.call_args.kwargs,dict(max_attempts_per_task=3,max_episodes=20,max_generated_tokens=1000))
                branch=json.loads((root/'out'/'branch.json').read_text())
                self.assertEqual(branch['starting_checkpoint_sha256'],record['checkpoint_sha256'])
                config.write_text(json.dumps(dict(mode='full',model_version='grpo-checkpoint')))
                with self.assertRaisesRegex(ValueError,'sampling config'):
                    collect_rs('manifest',config,checkpoint,export,root/'bad',max_attempts_per_task=3,max_episodes=20,max_generated_tokens=1000)
                self.assertEqual(collector.call_count,1)

    def test_optimizer_must_use_same_sft_origin_and_rs_dataset(self):
        import hashlib
        from officegym.training.sft import validate_plan
        from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
        from officegym.harness.versions import HARNESS_VERSION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint,export,record=self.source(root)
            (root/'samples.jsonl').write_text('fixture\n')
            manifest=dict(kind='sft_dataset_not_training_result',accepted=[{}],manual_alignment_review_status='passed',eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,harness_version=HARNESS_VERSION,
                          samples_sha256=hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest(),training_branch='rs_sft',source_export=str(export),source_export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),starting_checkpoint_sha256=record['checkpoint_sha256'])
            from review_fixtures import reviewed_fixture
            reviewed_fixture(root,manifest)
            plan=dict(training_branch='rs_sft',model_path=str(checkpoint),model_revision='fixture',max_steps=1,checkpoint_interval_steps=1,max_wall_seconds=60,
                      locked_packages={name:'fixture' for name in ('torch','transformers','trl','peft','datasets')},max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=0.0001,lora={})
            self.assertEqual(validate_plan(plan,root)['training_branch'],'rs_sft')
            with self.assertRaisesRegex(ValueError,'plan and dataset branch'):
                validate_plan({**plan,'training_branch':'sft'},root)
            (checkpoint/'model.safetensors').write_bytes(b'wrong-initial-weights')
            with self.assertRaisesRegex(ValueError,'unchanged exported SFT'):
                validate_plan(plan,root)

"""Preflight uses byte fixtures; no PEFT/model loading or real merge occurs."""
import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.merge import validate_merge_plan,worker
from officegym.training.checkpoints import checkpoint_fingerprint
from officegym.training.evidence import TrainingEvidence


class MergeTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'base';base.mkdir()
        (base/'config.json').write_text('{}')
        (base/'model.safetensors').write_bytes(b'fixture-not-a-model')
        adapter=root/'adapter';adapter.mkdir()
        (adapter/'tokenizer_config.json').write_text('{}')
        (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA'}))
        (adapter/'adapter_model.safetensors').write_bytes(b'fixture-not-an-adapter')
        TrainingEvidence(root).checkpoint(adapter,5,kind='final_adapter')
        return dict(base_model_path=str(base),base_revision='fixture',base_checkpoint_sha256=checkpoint_fingerprint(base)['checkpoint_sha256'],adapter_path=str(adapter),training_events=str(root/'training_events.jsonl'),optimizer_step=5,locked_packages={name:'fixture' for name in ('torch','transformers','peft')},device='cpu',dtype='float32',max_wall_seconds=60,max_shard_size_bytes=1000,max_export_bytes=10000)

    def test_source_and_finite_resource_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=self.fixture(root)
            self.assertEqual(validate_merge_plan(plan)['training_source']['optimizer_step'],5)
            for patch in ({'max_wall_seconds':float('inf')},{'max_export_bytes':0},{'max_shard_size_bytes':20000},{'base_model_path':'remote/model'},{'optimizer_step':6}):
                with self.subTest(patch=patch),self.assertRaises(ValueError):
                    validate_merge_plan({**plan,**patch})
            (Path(plan['base_model_path'])/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'content differs'):
                validate_merge_plan(plan)

    def test_worker_rejects_changed_plan_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed before execution'):
                worker(path,'wrong-checksum',directory)

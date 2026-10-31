"""Artifact checks use byte fixtures, not actual checkpoints or training curves."""
import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.evidence import TrainingEvidence
from officegym.training.sft import validate_plan


class TrainingEvidenceTests(unittest.TestCase):
    def test_all_logs_and_checkpoints_retained_with_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evidence=TrainingEvidence(root)
            evidence.event('training_log',1,{'loss':1.5})
            evidence.event('training_log',2,{'loss':float('nan')})
            hashes=[]
            for step in (1,2):
                checkpoint=root/('checkpoint-'+str(step));checkpoint.mkdir()
                (checkpoint/'adapter.safetensors').write_bytes(('fixture '+str(step)).encode())
                hashes.append(evidence.checkpoint(checkpoint,step)['checkpoint_files_sha256'])
            self.assertNotEqual(*hashes)
            events=[json.loads(line) for line in (root/'training_events.jsonl').read_text().splitlines()]
            self.assertEqual(len(events),4)
            self.assertEqual(events[1]['payload']['loss'],{'non_finite':'nan'})
            evidence.history([{'step':2,'loss':float('inf')}])
            self.assertEqual(json.loads((root/'trainer_log_history.json').read_text())[0]['loss'],{'non_finite':'inf'})

    def test_checkpoint_interval_required_and_bounded_before_model_io(self):
        plan=dict(model_path='missing',model_revision='fixture',max_steps=2,max_wall_seconds=60,locked_packages={},max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=.0001,lora={})
        with self.assertRaisesRegex(ValueError,'checkpoint_interval_steps'):
            validate_plan(plan,'missing')
        for value in (0,3,True):
            with self.assertRaisesRegex(ValueError,'checkpoint interval'):
                validate_plan({**plan,'checkpoint_interval_steps':value},'missing')

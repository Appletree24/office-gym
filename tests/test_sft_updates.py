"""Authored optimizer event fixtures never establish model learning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.evidence import TrainingEvidence
from officegym.training.sft_updates import UpdateRecorder,CONTRACT,verify_updates


def proof():return dict(contract=CONTRACT,gradient_tensors=2,gradient_parameters=3,trainable_parameters=4,gradient_l2_norm=.5)


class SFTUpdateTests(unittest.TestCase):
    def test_complete_steps_prefix_scope_and_zero_norm(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evidence=TrainingEvidence(root);updates=UpdateRecorder(evidence)
            for step in (1,2):
                value=proof();value['gradient_l2_norm']=0 if step==1 else .5
                updates.attempt(step,value);updates.finish(step,skipped=False);updates.step_end(step)
            path=root/'training_events.jsonl';prefix=path.stat().st_size
            self.assertEqual(verify_updates(path,prefix,2),2)
            updates.attempt(3,proof())
            self.assertEqual(verify_updates(path,prefix,2),2)
            with self.assertRaises(ValueError):verify_updates(path,path.stat().st_size,2)

    def test_pending_skip_and_nonfinite_fail_without_fake_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evidence=TrainingEvidence(root);updates=UpdateRecorder(evidence)
            bad=proof();bad['gradient_l2_norm']=float('nan')
            with self.assertRaises(ValueError):updates.attempt(1,bad)
            self.assertFalse((root/'training_events.jsonl').exists())
            updates.attempt(1,proof())
            with self.assertRaises(ValueError):updates.attempt(1,proof())
            with self.assertRaises(ValueError):updates.finish(1,skipped=True)
            path=root/'training_events.jsonl'
            self.assertNotIn('sft_optimizer_update',path.read_text())
            with self.assertRaises(ValueError):verify_updates(path,path.stat().st_size,1)

    def test_mutated_counts_order_skips_and_missing_attempts_reject(self):
        base=[dict(kind='sft_optimizer_attempt',optimizer_step=1,payload=proof()),dict(kind='sft_optimizer_update',optimizer_step=1,payload=dict(proof(),optimizer_step_was_skipped=False))]
        mutations=[lambda rows:rows.pop(0),lambda rows:rows.reverse(),lambda rows:rows[1]['payload'].update(gradient_l2_norm=.9),lambda rows:rows[1]['payload'].update(optimizer_step_was_skipped=0),lambda rows:rows[0]['payload'].update(gradient_parameters=99),lambda rows:rows[1].update(optimizer_step=2),lambda rows:rows.append(dict(kind='training_failed',optimizer_step=1,payload={}))]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                rows=copy.deepcopy(base);mutation(rows);path=Path(directory)/'events';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaises(ValueError):verify_updates(path,path.stat().st_size,1)

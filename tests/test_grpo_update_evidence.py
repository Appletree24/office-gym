"""Adversarial formal-export protocol fixtures; no model training occurs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.checkpoints import verify_grpo_updates


def records():
    counts=dict(selected_trajectories=2,actor_loss_tokens=4,forward_context_tokens=10,selected_generated_tokens=4,context_only_tokens=6)
    return [dict(kind='optimizer_update_attempt',optimizer_step=1,payload=dict(native_global_step=1,**counts)),
            dict(kind='optimizer_update',optimizer_step=1,payload=dict(native_global_step=1,contract='single-mini-batch-epoch',gradient_norms=[.5],native_actor_metrics={'actor/kl_loss':.001},sampled_actor_step=0,sampled_actor_version='admitted-sft',per_generation_version_checks=True,**counts))]


class UpdateEvidenceTests(unittest.TestCase):
    def verify(self,rows,**kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return verify_grpo_updates(path,path.stat().st_size,1,initial_actor='admitted-sft',expected_rows=2,**kwargs)

    def test_complete_bound_proof_and_attempt_pair(self):
        self.assertEqual(self.verify(records()),1)
        for rows in (records()[1:],records()[:1],[records()[0],*records()],list(reversed(records()))):
            with self.subTest(rows=rows),self.assertRaises(ValueError):self.verify(rows)

    def test_kl_mask_cost_and_actor_proof_cannot_be_removed_or_replaced(self):
        changes=[('native_actor_metrics',{}),('native_actor_metrics',{'actor/kl_loss':float('nan')}),
                 ('native_actor_metrics',{'actor/kl_loss':True}),('gradient_norms',[float('inf')]),
                 ('actor_loss_tokens',0),('selected_generated_tokens',5),('forward_context_tokens',11),
                 ('context_only_tokens',-1),('selected_trajectories',4),('actor_loss_tokens',True),
                 ('sampled_actor_step',1),('sampled_actor_version','another-sft'),('per_generation_version_checks',False)]
        for key,value in changes:
            rows=records();rows[1]['payload'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.verify(rows)
        for key in ('actor_loss_tokens','native_actor_metrics','sampled_actor_version'):
            rows=records();del rows[1]['payload'][key]
            with self.subTest(missing=key),self.assertRaises(ValueError):self.verify(rows)

    def test_unresolved_next_step_before_checkpoint_is_not_admissible(self):
        rows=records();next_attempt=deepcopy(rows[0]);next_attempt['optimizer_step']=2;next_attempt['payload']['native_global_step']=2
        with self.assertRaisesRegex(ValueError,'unresolved'):self.verify(rows+[next_attempt])

    def test_attempt_batch_must_match_verified_batch_even_when_totals_are_consistent(self):
        rows=records();rows[0]['payload'].update(actor_loss_tokens=5,selected_generated_tokens=5,context_only_tokens=5)
        with self.assertRaisesRegex(ValueError,'matching attempted batch'):self.verify(rows)

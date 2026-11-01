"""Authored accounting fixtures; no measured training/evaluation claims."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.curves import summarize,export


def sample(tokens,candidates,effective,status='refill'):
    return dict(kind='effective_group_sampling',optimizer_step=0,payload=dict(rollout_tokens=tokens,candidate_groups=candidates,effective_groups=effective,status=status,actor_version='fixture'))


class TrainingCurveTests(unittest.TestCase):
    def test_refill_counters_are_differenced_and_new_windows_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events'
            rows=[sample(100,2,0),sample(250,4,1,'insufficient_effective_groups'),sample(90,2,0)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            value=summarize(path)
            self.assertEqual([row['batch_generated_tokens'] for row in value['sampling_batches']],[100,150,90])
            self.assertEqual(value['totals']['reported_sampling_tokens'],340)
            self.assertEqual(value['totals']['candidate_groups'],6)
            self.assertEqual(value['totals']['protocol_verified_updates'],0)
            self.assertIsNone(value['evaluation_success_rates'])
            self.assertIsNotNone(value['open_sampling_window'])

    def test_sampling_end_does_not_double_count_unfinished_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events';row=sample(100,2,0)
            path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,kind='effective_group_sampling_end'))+'\n')
            self.assertEqual(summarize(path)['totals']['reported_sampling_tokens'],100)
            self.assertIsNone(summarize(path)['open_sampling_window'])

    def test_regressed_counters_and_inconsistent_ledger_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'events'
            path.write_text(json.dumps(sample(100,2,0))+'\n'+json.dumps(sample(90,4,0))+'\n')
            with self.assertRaises(ValueError):summarize(path)
            path.write_text(json.dumps(sample(100,2,0))+'\n')
            usage=root/'usage';usage.write_text(json.dumps(dict(actual_tokens_lower_bound=90,actual_tokens_upper_bound=200,observed_tokens_known=90,reserved_tokens=200)))
            with self.assertRaises(ValueError):export(path,root/'out',usage_path=usage)
            self.assertFalse((root/'out').exists())

    def test_unverified_attempt_is_retained_and_not_counted_as_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events';path.write_text(json.dumps(dict(kind='optimizer_update_attempt',optimizer_step=1,payload={'native_global_step':1}))+'\n')
            value=summarize(path)
            self.assertEqual(value['totals']['protocol_verified_updates'],0)
            self.assertEqual(value['unresolved_optimizer_attempt']['optimizer_step'],1)

    def test_complete_update_uses_protocol_gate_but_does_not_invent_sampling_cost(self):
        from test_grpo_update_evidence import records
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events';path.write_text(''.join(json.dumps(row)+'\n' for row in records()))
            value=summarize(path)
            self.assertEqual(value['totals']['protocol_verified_updates'],1)
            self.assertEqual(value['updates'][0]['cumulative_actor_loss_tokens'],4)
            self.assertEqual(value['updates'][0]['reference_kl'],.001)
            self.assertIsNone(value['ledger_usage'])
            self.assertFalse(value['formal_admission_present_but_not_verified'])
            self.assertIn('generation_cost_outside_recorded_sampling',value['costs_not_recorded_here'])

    def test_orphan_nonzero_sampling_end_is_not_silently_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events';path.write_text(json.dumps(dict(sample(100,2,0),kind='effective_group_sampling_end'))+'\n')
            with self.assertRaisesRegex(ValueError,'orphan sampling end'):summarize(path)

    def test_sampling_tool_costs_are_differenced_and_missing_counts_stay_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events'
            a=sample(100,2,0);b=sample(250,4,0,'insufficient_effective_groups')
            a['payload'].update(tool_calls_known=10,tool_trajectories_unknown=0)
            b['payload'].update(tool_calls_known=18,tool_trajectories_unknown=0)
            path.write_text(''.join(json.dumps(row)+'\n' for row in (a,b)))
            value=summarize(path)
            self.assertEqual([row['batch_tool_calls_known'] for row in value['sampling_batches']],[10,8])
            self.assertEqual(value['totals']['reported_sampling_tool_calls_known'],18)
            self.assertTrue(value['totals']['reported_sampling_tool_calls_exact'])
            b['payload']['tool_trajectories_unknown']=1
            path.write_text(''.join(json.dumps(row)+'\n' for row in (a,b)))
            self.assertFalse(summarize(path)['totals']['reported_sampling_tool_calls_exact'])
            with path.open('a') as stream:stream.write(json.dumps(sample(90,2,0))+'\n')
            value=summarize(path)
            self.assertEqual(value['totals']['reported_sampling_tool_calls_known'],18)
            self.assertIsNone(value['totals']['reported_sampling_tool_trajectories_unknown'])
            self.assertFalse(value['totals']['reported_sampling_tool_calls_exact'])

    def test_sampling_tool_counters_cannot_regress_or_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events';a=sample(100,2,0);a['payload'].update(tool_calls_known=10,tool_trajectories_unknown=0)
            for fields in ({'tool_calls_known':9,'tool_trajectories_unknown':0},{},{'tool_calls_known':True,'tool_trajectories_unknown':0}):
                b=sample(200,4,0);b['payload'].update(fields)
                path.write_text(''.join(json.dumps(row)+'\n' for row in (a,b)))
                with self.subTest(fields=fields),self.assertRaises(ValueError):summarize(path)

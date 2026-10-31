"""Numerical arrays are authored fixtures, not measurements of model equivalence."""
import tempfile
import unittest
from pathlib import Path
from officegym.training.equivalence import compare_logits,worker,validate


class NumericalEquivalenceTests(unittest.TestCase):
    def test_full_vector_tolerance_and_nonfinite_rejection(self):
        self.assertTrue(compare_logits([1.,2.],[1.001,2.002],.01)['within_tolerance'])
        self.assertFalse(compare_logits([1.,2.],[1.,3.],.01)['within_tolerance'])
        self.assertFalse(compare_logits([1.,float('nan')],[1.,2.],.01)['finite'])
        with self.assertRaises(ValueError):
            compare_logits([1.],[1.,2.],.01)
        with self.assertRaises(ValueError):
            compare_logits([1.],[1.],float('inf'))

    def test_changed_plan_rejected_before_optional_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=Path(directory)/'plan.json';plan.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed before execution'):
                worker(plan,'wrong-checksum',directory)
        with self.assertRaisesRegex(ValueError,'missing numerical'):
            validate({})

    def test_frozen_probe_and_allocation_limits_checked_without_loading_models(self):
        import json,hashlib
        from officegym.training.checkpoints import checkpoint_fingerprint
        from officegym.training.evidence import TrainingEvidence
        from officegym.training.merge import validate_merge_plan
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base=root/'base';base.mkdir()
            (base/'config.json').write_text('{"vocab_size":4}')
            (base/'model.safetensors').write_bytes(b'fixture-not-a-model')
            adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{"peft_type":"LORA"}')
            (adapter/'tokenizer_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture-not-an-adapter')
            TrainingEvidence(root).checkpoint(adapter,1)
            merged=root/'merged';merged.mkdir()
            (merged/'config.json').write_text('{"vocab_size":4}')
            (merged/'model.safetensors').write_bytes(b'fixture-not-merged-weights')
            merge_plan=dict(base_model_path=str(base),base_revision='fixture',base_checkpoint_sha256=checkpoint_fingerprint(base)['checkpoint_sha256'],adapter_path=str(adapter),training_events=str(root/'training_events.jsonl'),optimizer_step=1,locked_packages={name:'fixture' for name in ('torch','transformers','peft')},device='cpu',dtype='float32',max_wall_seconds=60,max_shard_size_bytes=100,max_export_bytes=10000)
            merge_plan_path=root/'merge_plan.json';merge_plan_path.write_text(json.dumps(merge_plan))
            merged_record=dict(validate_merge_plan(merge_plan),merged=True,model_path=str(merged),exported_checkpoint_sha256=checkpoint_fingerprint(merged)['checkpoint_sha256'])
            merged_record_path=root/'merge_result.json';merged_record_path.write_text(json.dumps(merged_record))
            probes=root/'probes.json';probes.write_text(json.dumps([{'case_id':'fixture','input_ids':[1,2]}]))
            plan=dict(merge_plan=str(merge_plan_path),merge_plan_sha256=sha(merge_plan_path),merge_result=str(merged_record_path),merge_result_sha256=sha(merged_record_path),probes=str(probes),probes_sha256=sha(probes),logit_atol=.01,min_generation_match_rate=1.,max_cases=1,max_input_tokens=2,max_new_tokens=2,max_wall_seconds=60,max_probe_logits_elements=8)
            self.assertEqual(len(validate(plan)[2]),1)
            with self.assertRaisesRegex(ValueError,'allocation exceeds'):
                validate({**plan,'max_probe_logits_elements':1})
            with self.assertRaisesRegex(ValueError,'excessive fixed probe'):
                validate({**plan,'max_input_tokens':1})
            probes.write_text(json.dumps([{'case_id':'fixture','input_ids':[4]}]))
            with self.assertRaisesRegex(ValueError,'input hash/path'):
                validate(plan)
            with self.assertRaisesRegex(ValueError,'invalid or excessive'):
                validate({**plan,'probes_sha256':sha(probes)})

    def test_relative_tolerance_is_elementwise_and_reference_anchored(self):
        result=compare_logits([100.,0.],[100.0005,.00005],1e-4,1e-5)
        self.assertTrue(result['within_tolerance'])
        self.assertGreater(result['max_abs_error'],1e-4)
        # A large unrelated coordinate cannot excuse a bad small coordinate.
        self.assertFalse(compare_logits([100.,0.],[100.,.0002],1e-4,1e-5)['within_tolerance'])
        self.assertFalse(compare_logits([100.],[100.0005],1e-4)['within_tolerance'])
        self.assertTrue(compare_logits([1.],[1.],0)['within_tolerance'])
        self.assertFalse(compare_logits([1.],[2.],0)['within_tolerance'])
        for rtol in (True,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):compare_logits([1.],[1.],.01,rtol)

    def test_replay_rejects_forged_metrics_and_rehashed_vector_changes(self):
        import json,copy
        from officegym.training.equivalence import replay_comparisons,replay_control,digest,NUMERIC_REPORT_VERSION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);probes=[dict(case_id='fixture',input_ids=[0])];sides={}
            for side,logits in [('before',[1.,2.]),('after',[1.00001,2.]),('control',[2.,3.])]:
                path=root/(side+'.json');path.write_text(json.dumps(dict(input_ids=[0],generated_ids=[1],next_token_logits=logits)))
                sides[side]=[dict(case_id='fixture',artifact=str(path),sha256=digest(path),generated_ids=[1])]
            checks=[dict(case_id='fixture',logits=compare_logits([1.,2.],[1.00001,2.],1e-4,1e-5),generation_equal=True)]
            report=dict(numerical_report_version=NUMERIC_REPORT_VERSION,before=sides['before'],after=sides['after'],comparisons=checks)
            self.assertEqual(replay_comparisons(report,probes,2,1e-4,1e-5),checks)
            altered=copy.deepcopy(report);altered['comparisons'][0]['logits']['max_abs_error']=0
            with self.assertRaisesRegex(ValueError,'raw artifact replay'):replay_comparisons(altered,probes,2,1e-4,1e-5)
            path=Path(sides['after'][0]['artifact']);path.write_text(json.dumps(dict(input_ids=[0],generated_ids=[1],next_token_logits=[2.,3.])))
            report['after'][0]['sha256']=digest(path)
            with self.assertRaisesRegex(ValueError,'raw artifact replay'):replay_comparisons(report,probes,2,1e-4,1e-5)
            control=dict(kind='disabled_adapter',detected=True,records=sides['control'],comparisons=[dict(case_id='fixture',logits=compare_logits([1.,2.],[2.,3.],1e-4,1e-5),generation_equal=True)])
            report['negative_control']=control
            self.assertTrue(replay_control(report,probes,2,1e-4,1e-5,'disabled_adapter'))

    def test_replay_preserves_strict_original_and_generated_id_types(self):
        import json
        from officegym.training.equivalence import replay_comparisons,digest,NUMERIC_REPORT_VERSION
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.json'
            for input_ids,generated_ids in [([False],[1]),([0],[]),([0],[True])]:
                path.write_text(json.dumps(dict(input_ids=input_ids,generated_ids=generated_ids,next_token_logits=[1.,2.])))
                row=dict(case_id='fixture',artifact=str(path),sha256=digest(path),generated_ids=generated_ids)
                report=dict(numerical_report_version=NUMERIC_REPORT_VERSION,before=[row],after=[row],comparisons=[])
                with self.assertRaises(ValueError):replay_comparisons(report,[dict(case_id='fixture',input_ids=[0])],2,.01,0)

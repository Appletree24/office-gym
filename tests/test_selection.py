"""Selection tests fabricate all weights/results; these are not experiment evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from officegym.eval.selection import freeze,select
from officegym.eval.plan import freeze_plan
from officegym.training.checkpoints import checkpoint_fingerprint,record_export
from officegym.training.evidence import TrainingEvidence


class SelectionTests(unittest.TestCase):
    def fixture(self,root):
        plan=[dict(group='sft_full',split='dev',instance_id='fixture',repeat_id=repeat,scenario_family_id='family',template_id='D01',depth_level=1,perturbation_level=0,dependency_width=1) for repeat in range(3)]
        freeze_plan(plan,root/'plan.jsonl')
        candidates=[];files={}
        evidence=TrainingEvidence(root)
        for index,step in enumerate((10,5)):
            checkpoint=root/('checkpoint'+str(index));checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(('fixture-'+str(index)).encode())
            fingerprint=checkpoint_fingerprint(checkpoint)['checkpoint_sha256']
            export=root/('export'+str(index)+'.json')
            inventory=evidence.checkpoint(checkpoint,step)
            equivalence=root/('equivalence'+str(index)+'.json')
            equivalence.write_text(json.dumps(dict(passed=True,checks=['authored-fixture-not-real-equivalence'],exported_checkpoint_sha256=fingerprint,source_checkpoint_files_sha256=inventory['checkpoint_files_sha256'],optimizer_step=step)))
            record_export(checkpoint,export,stage='sft',parent_hash='fixture',training_manifest_hash='fixture',versions={},equivalence_report=equivalence,training_events=root/'training_events.jsonl',source_checkpoint=checkpoint,optimizer_step=step)
            candidates.append(dict(candidate_id=str(index),optimizer_step=step,checkpoint=str(checkpoint),export_record=str(export)))
            results=[]
            for planned in plan:
                results.append(dict(planned,context_metrics_version='context-metrics-v1',context_samples=1,context_token_sum=20,peak_context_tokens=20,compression_trigger_count=0,summary_model_calls=0,summary_cache_hits=0,compressed_context_count=0,truncated_context_count=0,model_version=fingerprint,env_version='fixture',eval_grader_version='fixture',reward_version='fixture',harness_version='fixture',dataset_manifest_hash='fixture',budget_hash='fixture',success=True,infra_error=False,fault_injected=False,recovery_started=False,duplicate_effects=0,premature_completion=False,declared_done=True,generated_tokens=1,input_tokens=1,tool_calls=1,wall_seconds=1,sim_seconds=30))
            path=root/('results'+str(index)+'.jsonl')
            path.write_text(''.join(json.dumps(row)+'\n' for row in results));files[str(index)]=str(path)
        freeze(candidates,root/'plan.jsonl','sft_full',root/'rule.json')
        return candidates,files

    def test_dev_tie_selects_earlier_step_and_retains_all_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,files=self.fixture(root)
            result=select(root/'rule.json',files,root/'selected.json')
            self.assertEqual(result['selected']['optimizer_step'],5)
            self.assertEqual(len(result['all_candidates']),2)
            self.assertEqual(result['selected']['dev_macro_success'],1)

    def test_missing_results_and_test_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,files=self.fixture(root)
            path=Path(files['0']);original=path.read_text()
            path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'missing dev runs'):
                select(root/'rule.json',files,root/'selected.json')
            altered=[json.loads(line) for line in original.splitlines()]
            altered[0]['split']='id_test'
            path.write_text(''.join(json.dumps(row)+'\n' for row in altered))
            with self.assertRaisesRegex(ValueError,'must match dev'):
                select(root/'rule.json',files,root/'selected.json')

    def test_changed_candidate_and_incomparable_budgets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);candidates,files=self.fixture(root)
            path=Path(files['0'])
            altered=[json.loads(line) for line in path.read_text().splitlines()]
            altered[0]['budget_hash']='different'
            path.write_text(''.join(json.dumps(row)+'\n' for row in altered))
            with self.assertRaisesRegex(ValueError,'incomparable dev'):
                select(root/'rule.json',files,root/'selected.json')
            (Path(candidates[0]['checkpoint'])/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'candidate changed'):
                select(root/'rule.json',files,root/'selected.json')

    def test_test_plan_is_rejected_before_any_checkpoint_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            freeze_plan([dict(group='sft_full',split='id_test')],root/'plan.jsonl')
            with self.assertRaisesRegex(ValueError,'dev-only'):
                freeze([],root/'plan.jsonl','sft_full',root/'rule.json')

    def test_candidate_step_cannot_override_recorded_training_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);candidates,_=self.fixture(root)
            candidates[0]['optimizer_step']=99
            with self.assertRaisesRegex(ValueError,'optimizer step differs'):
                freeze(candidates,root/'plan.jsonl','sft_full',root/'changed_rule.json')

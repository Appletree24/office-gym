"""Selection tests fabricate all weights/results; these are not experiment evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from officegym.eval.selection import freeze,select
from officegym.eval.plan import freeze_plan
from officegym.training.checkpoints import checkpoint_fingerprint


class SelectionTests(unittest.TestCase):
    def fixture(self,root):
        plan=[dict(group='sft_full',split='dev',instance_id='fixture',repeat_id=repeat,scenario_family_id='family',template_id='D01',depth_level=1,perturbation_level=0) for repeat in range(3)]
        freeze_plan(plan,root/'plan.jsonl')
        candidates=[];files={}
        for index,step in enumerate((10,5)):
            checkpoint=root/('checkpoint'+str(index));checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(('fixture-'+str(index)).encode())
            fingerprint=checkpoint_fingerprint(checkpoint)['checkpoint_sha256']
            export=root/('export'+str(index)+'.json')
            export.write_text(json.dumps(dict(stage='sft',checkpoint_sha256=fingerprint,equivalence_report_sha256='fixture-not-real-equivalence')))
            candidates.append(dict(candidate_id=str(index),optimizer_step=step,checkpoint=str(checkpoint),export_record=str(export)))
            results=[]
            for planned in plan:
                results.append(dict(planned,model_version=fingerprint,env_version='fixture',eval_grader_version='fixture',reward_version='fixture',harness_version='fixture',dataset_manifest_hash='fixture',budget_hash='fixture',success=True,infra_error=False,fault_injected=False,recovery_started=False,duplicate_effects=0,premature_completion=False,declared_done=True,generated_tokens=1,input_tokens=1,tool_calls=1,wall_seconds=1,sim_seconds=30))
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

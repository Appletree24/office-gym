import copy
import json
from pathlib import Path
import tempfile
import unittest
from officegym.tasks.generate import generate
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.eval.grading import grade_episode_snapshot,require_recorded_grade


class EpisodeGradeEvidenceTests(unittest.TestCase):
    def test_valid_business_world_cannot_make_budget_stop_success_in_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T01'],count_per_template=1);source=root/'tasks'/manifest['instances'][0]['path'];task=json.loads((source/'task.json').read_text());spec={**task['public_task'],**json.loads((source/'private_spec.json').read_text())}
            world=clone_world(source/'initial_world.sqlite',root/'world.sqlite')
            try:solve(task['public_task'],Dispatcher(world).call)
            finally:world.close()
            for state in ('DECLARED_DONE','BUDGET_STOP','PARSE_ERROR','CONTEXT_OVERFLOW','INFRA_ERROR'):
                grade,audit=grade_episode_snapshot(root/'world.sqlite',spec,source/'initial_world.sqlite',termination_state=state)
                result=dict(termination={'state':state},grade=grade,operation_audit=audit,infrastructure_error=None)
                self.assertEqual(require_recorded_grade(root/'world.sqlite',spec,source/'initial_world.sqlite',result),(grade,audit))
                self.assertTrue(grade['outcome_ok']);self.assertEqual(grade['full_success'],state=='DECLARED_DONE')
                if state!='DECLARED_DONE':
                    forged=copy.deepcopy(result);forged['grade'].update(full_success=True,reward=1.0)
                    with self.assertRaisesRegex(ValueError,'recorded grade differs'):require_recorded_grade(root/'world.sqlite',spec,source/'initial_world.sqlite',forged)
            forged=copy.deepcopy(result);forged['operation_audit']['operations']+=1
            with self.assertRaisesRegex(ValueError,'operation audit differs'):require_recorded_grade(root/'world.sqlite',spec,source/'initial_world.sqlite',forged)
            forged=copy.deepcopy(result);forged['grade']['completion_checks']['runtime_accepted']=True
            with self.assertRaisesRegex(ValueError,'recorded grade differs'):require_recorded_grade(root/'world.sqlite',spec,source/'initial_world.sqlite',forged)
            with self.assertRaisesRegex(ValueError,'terminal Harness'):grade_episode_snapshot(root/'world.sqlite',spec,source/'initial_world.sqlite',termination_state='READY')

    def test_infrastructure_error_rejects_even_declared_done_with_valid_world(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T01'],count_per_template=1);source=root/'tasks'/manifest['instances'][0]['path'];task=json.loads((source/'task.json').read_text());spec={**task['public_task'],**json.loads((source/'private_spec.json').read_text())}
            grade,audit=grade_episode_snapshot(source/'reference_final.sqlite',spec,source/'initial_world.sqlite',termination_state='DECLARED_DONE',infrastructure_error={'type':'AuthoredInfrastructureFailure'})
            self.assertTrue(grade['outcome_ok']);self.assertTrue(grade['declared_done']);self.assertFalse(grade['full_success']);self.assertFalse(grade['completion_checks']['runtime_accepted']);self.assertEqual(grade['reward'],0.0)

    def test_export_entrypoint_rejects_altered_grade_after_real_fixture_execution(self):
        from dataclasses import asdict
        from unittest.mock import patch
        from officegym.eval.episode import run_episode
        from officegym.eval.export import result_row
        from officegym.harness.loop import AssistantTurn,Budget
        class ReferenceFixture:
            model_version='authored-export-contract-fixture'
            def __init__(self,actions):self.actions=iter(actions)
            def generate(self,*args):
                action=next(self.actions);return AssistantTurn(action['tool'],action['arguments'],5,3)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['D01'],count_per_template=1,split='dev');entry=manifest['instances'][0];source=root/'tasks'/entry['path']
            actions=[json.loads(line) for line in (source/'reference_trace.jsonl').read_text().splitlines()]
            # Only this test bypasses independent model/isolation/context gates to reach
            # the export contract with real business evidence. No inference is claimed.
            result=run_episode(source,root/'run',ReferenceFixture(actions),mode='full')
            self.assertTrue(result['grade']['full_success'])
            result['policy_kind']='inference_service'
            (root/'run/result.json').write_text(json.dumps(result))
            config=dict(budget=asdict(Budget()),max_context_tokens=8192,max_output_tokens=1024)
            (root/'run/resolved_eval_config.json').write_text(json.dumps(config))
            planned={key:entry[key] for key in ('instance_id','split','template_id','scenario_family_id','depth_level','perturbation_level','dependency_width')};planned.update(group='base_full',repeat_id=0)
            with patch('officegym.harness.isolation.require_isolation'), patch('officegym.eval.export.context_metrics',return_value={}):
                row=result_row(root/'run',planned,root/'tasks/manifest.json')
                self.assertTrue(row['success']);self.assertEqual(row['terminated_reason'],'DECLARED_DONE');self.assertEqual(row['completion_checks'],result['grade']['completion_checks'])
                result['grade']['full_success']=False
                (root/'run/result.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'recorded grade differs'):result_row(root/'run',planned,root/'tasks/manifest.json')

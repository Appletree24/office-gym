import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.training.environment import RolloutEnvironment


class RLEnvironmentTests(unittest.TestCase):
    def test_independent_worlds_grade_before_release_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source = root/'tasks'/manifest['instances'][0]['path']
            with RolloutEnvironment(source,actor_version='sft0',work_dir=root,artifact_dir=root/'archive') as first, RolloutEnvironment(source,actor_version='sft0',work_dir=root) as second:
                first.call('board.patch',{'expected_version':1,'patch':{'goal':'modified'}})
                self.assertEqual(second.call('board.read',{})['data']['version'],1)
                result = first.finalize(input_ids=[1,2],response_mask=[1,0],token_roles=['assistant','tool'],logprobs=[-.1,-.2],actor_version='sft0')
                self.assertEqual(result['reward'],0)
                self.assertEqual(result['original_token_ids'],[1,2])
                temporary = first._path
                with self.assertRaises(ValueError):
                    first.call('board.read',{})
            self.assertFalse(temporary.exists())
            self.assertTrue((root/'archive'/'terminal.json').is_file())
            self.assertTrue((root/'archive'/'final_world.sqlite').is_file())
            self.assertTrue((root/'archive'/'tools.jsonl').is_file())

    def test_budget_failures_archive_the_same_reward_returned_to_optimizer(self):
        import json
        from officegym.tasks.reference import solve
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source=root/'tasks'/manifest['instances'][0]['path']
            for reason in ('policy_terminated','response_limit','generation_budget','invalid_tool_batch','agent_turn_limit'):
                archive=root/reason
                with RolloutEnvironment(source,actor_version='fixture',artifact_dir=archive) as environment:
                    solve(environment.public_task,lambda name,args,operation=None:environment.call(name,args))
                    result=environment.finalize(input_ids=[1,2],response_mask=[1,0],token_roles=['assistant','tool'],logprobs=[-.1,0.0],actor_version='fixture',termination_reason=reason)
                    self.assertTrue(result['domain_grade']['full_success'])
                    self.assertEqual(result['domain_grade']['terminated_reason'],'DECLARED_DONE')
                    self.assertEqual(result['domain_grade']['termination_reason_source'],'world_snapshot')
                    self.assertEqual(result['grade']['terminated_reason'],reason)
                    self.assertEqual(result['grade']['termination_reason_source'],'native_rollout')
                    self.assertEqual(result['grade']['completion_checks']['runtime_accepted'],reason=='policy_terminated')
                    self.assertTrue(result['operation_audit']['consistent'])
                    self.assertEqual(result['grade']['full_success'],reason=='policy_terminated')
                    self.assertEqual(result['reward'],1.0 if reason=='policy_terminated' else 0.0)
                    self.assertEqual(json.loads((archive/'terminal.json').read_text()),result)

    def test_terminal_publish_failure_leaves_explicit_incomplete_world(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source=root/'tasks'/manifest['instances'][0]['path']
            environment=RolloutEnvironment(source,actor_version='fixture',artifact_dir=root/'archive')
            with patch('officegym.training.environment.write_json',side_effect=OSError('fixture publish failure')):
                with self.assertRaises(OSError):
                    environment.finalize(input_ids=[1],response_mask=[1],token_roles=['assistant'],logprobs=[-.1],actor_version='fixture')
            environment.close()
            self.assertTrue((root/'archive'/'final_world.sqlite').is_file())
            self.assertTrue((root/'archive'/'incomplete.json').is_file())
            self.assertFalse((root/'archive'/'terminal.json').exists())

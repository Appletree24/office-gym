import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.eval.episode import run_episode
from officegym.harness.loop import AssistantTurn,Budget


class EpisodeTests(unittest.TestCase):
    def test_early_finish_not_success_and_artifacts_retained(self):
        self_outer = self
        class Policy:
            model_version = 'scripted-unit-fixture'
            def generate(self,*args):
                self_outer.assertTrue((root/'run'/'resolved_eval_config.json').is_file())
                self_outer.assertTrue((root/'run'/'tokenizer_manifest.json').is_file())
                return AssistantTurn('task.finish',{'summary':'done'},5,3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1)
            result = run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',Policy(),mode='simple',initial_metadata={'resolved_eval_config.json':{'fixture':True},'tokenizer_manifest.json':{'fixture':True}})
            self.assertEqual(result['termination']['state'],'DECLARED_DONE')
            self.assertEqual(result['grade']['terminated_reason'],'DECLARED_DONE')
            self.assertEqual(result['grade']['termination_reason_source'],'harness_runtime')
            self.assertTrue(result['grade']['completion_checks']['runtime_accepted'])
            self.assertFalse(result['grade']['full_success'])
            self.assertTrue((root/'run'/'trajectory.jsonl').is_file())
            self.assertTrue((root/'run'/'world.sqlite').is_file())
            self.assertTrue((root/'run'/'result.json').is_file())
            self.assertTrue((root/'run'/'provenance.json').is_file())
            self.assertEqual(sum(result['versions']['tool_counts'].values()),24)
            self.assertEqual(result['versions']['model_version'],'scripted-unit-fixture')

    def test_custom_call_budget_injected_before_service(self):
        class Policy:
            model_version = 'scripted-fixture'
            def generate(self,*args):
                return AssistantTurn('board.read',{},5,3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1)
            result = run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',Policy(),mode='full',budget=Budget(tool_calls=2))
            self.assertEqual(result['termination']['state'],'BUDGET_STOP')
            self.assertEqual(result['grade']['terminated_reason'],'BUDGET_STOP')
            self.assertFalse(result['grade']['completion_checks']['runtime_accepted'])
            self.assertTrue(result['grade']['completion_checks']['within_simulated_deadline'])
            self.assertEqual(result['termination']['tool_calls'],2)
            self.assertIsNone(result['infrastructure_error'])
            self.assertTrue((root/'run'/'budget.json').is_file())

    def test_long_utf8_output_path_uses_short_private_socket_and_keeps_artifacts(self):
        class Policy:
            model_version='scripted-long-path-fixture'
            def generate(self,*args):return AssistantTurn('task.finish',{'summary':'done'},5,3)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            out=root/('长路径'*20)/'retained_episode'
            self.assertGreater(len(str(out/'world.sock').encode()),108)
            result=run_episode(root/'tasks'/manifest['instances'][0]['path'],out,Policy(),mode='full')
            self.assertEqual(result['termination']['state'],'DECLARED_DONE');self.assertIsNone(result['infrastructure_error'])
            self.assertTrue((out/'world.sqlite').is_file());self.assertTrue((out/'runtime.sqlite').is_file());self.assertTrue((out/'result.json').is_file())
            self.assertFalse((out/'world.sock').exists())

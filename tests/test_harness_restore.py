import tempfile
import unittest
from pathlib import Path
from officegym.harness.loop import Harness,AssistantTurn,Budget


class RestoreTests(unittest.TestCase):
    def test_usage_and_complete_interactions_restored(self):
        class Policy:
            calls = 0
            def generate(self,*args):
                self.calls += 1
                if self.calls==2:
                    raise ConnectionError('simulated infrastructure error')
                return AssistantTurn('board.read',{},5,3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace'
            budget = Budget(tool_calls=1)
            first = Harness(Policy(),lambda *args:{'ok':True,'data':{},'error':None},{},path,budget)
            self.assertEqual(first.run()['state'],'BUDGET_STOP')
            restored = Harness(Policy(),lambda *args:self.fail('budget was reset'),{},path,budget)
            usage = restored.restore()
            self.assertEqual(usage['tool_calls'],1)
            self.assertEqual(usage['generated_tokens'],3)
            self.assertEqual(len(restored.messages),4)
            self.assertEqual(restored.run()['state'],'BUDGET_STOP')
            with self.assertRaises(ValueError):
                Harness(Policy(),None,{'different':True},path,budget).restore()

    def test_elapsed_usage_survives_missing_termination_record(self):
        import json
        class Policy:
            def generate(self,*args):
                return AssistantTurn('board.read',{},5,3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace'
            budget = Budget(tool_calls=1)
            first = Harness(Policy(),lambda *args:{'ok':True,'data':{},'error':None},{},path,budget)
            first.run()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [row for row in rows if row['kind']!='termination']
            rows[-1]['elapsed_wall_seconds'] = 12.5
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            restored = Harness(Policy(),None,{},path,budget)
            self.assertEqual(restored.restore()['wall_seconds'],12.5)

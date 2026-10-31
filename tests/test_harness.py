import tempfile
import unittest
from pathlib import Path
from officegym.harness.loop import Harness, Budget, AssistantTurn


class HarnessTests(unittest.TestCase):
    def test_budget_stops_before_action(self):
        class Policy:
            def generate(self,*args):
                return AssistantTurn('mail.send',{},10,11)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(Policy(),lambda *args:calls.append(args),{},Path(directory)/'trace',Budget(generated_tokens=10))
            self.assertEqual(harness.run()['state'],'BUDGET_STOP')
            self.assertEqual(calls,[])

    def test_finish_is_declaration_only(self):
        class Policy:
            def generate(self,*args):
                return AssistantTurn('task.finish',{'summary':'done'},10,4)
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(Policy(),lambda *args: {'ok':True,'data':{},'error':None}, {},Path(directory)/'trace')
            result = harness.run()
            self.assertEqual(result['state'],'DECLARED_DONE')
            self.assertNotIn('full_success',result)

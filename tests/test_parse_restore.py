import tempfile
import unittest
from pathlib import Path
from officegym.harness.loop import Harness,AssistantTurn,Budget


class ParseRestoreTests(unittest.TestCase):
    def test_parse_failure_retries_not_reset(self):
        class BrokenThenUnavailable:
            calls = 0
            def generate(self,*args):
                self.calls += 1
                if self.calls==2:
                    raise ConnectionError('outage')
                return AssistantTurn(None,None,4,2)
        class Broken:
            def generate(self,*args):
                return AssistantTurn(None,None,4,2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace'
            first = Harness(BrokenThenUnavailable(),None,{},path,Budget(parse_retries=1))
            self.assertEqual(first.run()['state'],'INFRA_ERROR')
            second = Harness(Broken(),None,{},path,Budget(parse_retries=1))
            second.restore()
            self.assertEqual(second.parse_failures,1)
            self.assertEqual(len(second.messages),3)
            self.assertEqual(second.run()['state'],'PARSE_ERROR')

    def test_unbounded_budgets_rejected(self):
        with self.assertRaises(ValueError):
            Budget(wall_seconds=float('inf'))

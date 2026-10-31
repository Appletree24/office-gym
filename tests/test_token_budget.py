import tempfile
import unittest
from pathlib import Path
from officegym.training.token_budget import TokenLedger


class TokenBudgetTests(unittest.TestCase):
    def test_workers_share_reservations_and_crash_does_not_refund(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'budget.sqlite'
            first,second = TokenLedger(path,100),TokenLedger(path,100)
            first.reserve('a','actor0',60)
            first.close()
            with self.assertRaisesRegex(ValueError,'exhausted'):
                second.reserve('b','actor0',60)
            second.reserve('b','actor0',40)
            second.record('b',30)
            with self.assertRaises(ValueError):
                second.record('b',41)
            second.close()
            with self.assertRaises(ValueError):
                TokenLedger(path,200)

    def test_per_turn_generation_and_context_caps(self):
        from officegym.training.token_budget import GenerationBudget
        budget = GenerationBudget(100,200)
        self.assertEqual(budget.next_cap(150,80),50)
        budget.record(40,50)
        self.assertEqual(budget.next_cap(100,80),60)
        budget.record(60,60)
        self.assertEqual(budget.next_cap(100,80),0)
        self.assertTrue(budget.stopped)
        with self.assertRaises(ValueError):
            GenerationBudget(100,200).record(51,50)

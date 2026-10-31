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

    def test_usage_bounds_include_unresolved_failed_rollouts_and_actor_versions(self):
        from officegym.training.token_budget import create_token_ledger,read_token_usage
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tokens.sqlite';create_token_ledger(path,100)
            ledger=TokenLedger(path,100)
            ledger.reserve('completed','actor0',40);ledger.record('completed',17)
            ledger.reserve('interrupted','actor1',60);ledger.close()
            usage=read_token_usage(path,100)
            self.assertEqual(usage['reserved_tokens'],100)
            self.assertEqual((usage['actual_tokens_lower_bound'],usage['actual_tokens_upper_bound']),(17,77))
            self.assertFalse(usage['actual_tokens_exact'])
            self.assertEqual(usage['by_actor']['actor0']['actual_tokens_upper_bound'],17)
            self.assertEqual(usage['by_actor']['actor1']['unresolved_trajectories'],1)
            with self.assertRaises(FileExistsError):create_token_ledger(path,100)
            self.assertEqual(read_token_usage(path,100),usage)
            with self.assertRaisesRegex(ValueError,'budget differs'):read_token_usage(path,99)
            ledger=TokenLedger(path,100);ledger.record('interrupted',9);ledger.close()
            usage=read_token_usage(path,100)
            self.assertTrue(usage['actual_tokens_exact'])
            self.assertEqual((usage['actual_tokens_lower_bound'],usage['actual_tokens_upper_bound']),(26,26))
            self.assertEqual(usage['reserved_tokens'],100)

    def test_readonly_usage_does_not_create_missing_ledger(self):
        import sqlite3
        from officegym.training.token_budget import read_token_usage
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'missing.sqlite'
            with self.assertRaises(sqlite3.OperationalError):read_token_usage(path,100)
            self.assertFalse(path.exists())

    def test_committed_reservation_survives_worker_process_crash(self):
        import subprocess,sys
        from officegym.training.token_budget import create_token_ledger,read_token_usage
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tokens.sqlite';create_token_ledger(path,100)
            command=[sys.executable,'-c',
                     "import os,sys;from officegym.training.token_budget import TokenLedger;ledger=TokenLedger(sys.argv[1],100);ledger.reserve('crashed','actor0',60);os._exit(7)",str(path)]
            result=subprocess.run(command,timeout=10,check=False)
            self.assertEqual(result.returncode,7)
            usage=read_token_usage(path,100)
            self.assertEqual((usage['actual_tokens_lower_bound'],usage['actual_tokens_upper_bound']),(0,60))
            self.assertEqual(usage['unresolved_trajectories'],1)
            self.assertFalse(usage['actual_tokens_exact'])

    def test_archived_wal_ledger_is_portable_and_not_replaced(self):
        import hashlib
        from officegym.training.token_budget import create_token_ledger,archive_token_ledger,read_token_usage
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'live.sqlite';copy=Path(directory)/'archived.sqlite'
            create_token_ledger(path,100);ledger=TokenLedger(path,100)
            ledger.db.execute('PRAGMA journal_mode=WAL')
            ledger.reserve('a','actor0',60);ledger.record('a',17)
            expected=read_token_usage(path,100)
            checksum=archive_token_ledger(path,copy)
            self.assertEqual(checksum,hashlib.sha256(copy.read_bytes()).hexdigest())
            ledger.reserve('b','actor1',40);ledger.close()
            self.assertEqual(read_token_usage(copy,100),expected)
            self.assertNotEqual(read_token_usage(path,100),expected)
            with self.assertRaises(FileExistsError):archive_token_ledger(path,copy)
            self.assertEqual(read_token_usage(copy,100),expected)

import tempfile
from pathlib import Path
import unittest
from officegym.env.lifecycle_probe import run


class LifecycleProbeTests(unittest.TestCase):
    def test_superseded_and_cancelled_end_and_minutes_events_do_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'probe';report=run(root)
            self.assertTrue(report['all_checks_passed']);self.assertEqual(report['case_executions'],6)
            cases={row['scenario']:row for row in report['scenarios']}
            self.assertEqual((cases['complete']['meeting_ended'],cases['complete']['minutes_available']),(1,1))
            self.assertEqual((cases['cancel_before_end']['meeting_ended'],cases['cancel_before_end']['minutes_available']),(0,0))
            self.assertEqual((cases['cancel_before_minutes']['meeting_ended'],cases['cancel_before_minutes']['minutes_available']),(1,0))
            self.assertTrue(all(row['both_replays_equal'] for row in cases.values()))
            with self.assertRaises(FileExistsError):run(root)

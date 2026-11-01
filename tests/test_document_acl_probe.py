from pathlib import Path
import tempfile
import unittest
from officegym.env.document_acl_probe import run


class DocumentACLProbeTests(unittest.TestCase):
    def test_read_only_immutable_versions_atomic_denial_and_replay_after_new_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'probe';report=run(root)
            self.assertTrue(report['all_checks_passed']);self.assertTrue(report['business_and_observation_replays_equal'])
            self.assertEqual(report['executions'],2);self.assertEqual(report['denied_calls_per_execution'],8)
            with self.assertRaises(FileExistsError):run(root)

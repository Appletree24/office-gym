import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.reply_evidence_probe import run


class ReplyEvidenceProbeTests(unittest.TestCase):
    def test_real_sender_and_request_provenance_with_deterministic_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'probe';report=run(root)
            self.assertTrue(report['passed']);self.assertEqual(report['executions'],6);self.assertEqual(report['deterministic_pairs'],3);self.assertEqual(report['model_calls'],0)
            for scenario in ('genuine','self_spoof','clarification'):
                grade=json.loads((root/(scenario+'_1')/'grade.json').read_text())
                self.assertEqual(grade['full_success'],scenario=='genuine')
                if scenario!='genuine':
                    self.assertIn('invitation_before_valid_availability_reply',grade['hard_violations'])
                    self.assertNotIn('wrong_recipient',grade['hard_violations'])
                    self.assertNotIn('operation_evidence_inconsistent',grade['hard_violations'])
            with self.assertRaises(FileExistsError):run(root)

import tempfile
import unittest
from pathlib import Path
from officegym.training.audit import append_finding,summarize


class AuditTests(unittest.TestCase):
    def test_authored_cases_excluded_and_root_causes_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'actions.jsonl').write_text('{}\n')
            (root/'repro.json').write_text('{}')
            for identity in ('a','b'):
                append_finding(root/'audit.jsonl',finding_id=identity,root_cause='duplicate_send',source_kind='authored_counterexample',source_run=root,minimal_reproduction=root/'repro.json',old_grade={'eval_grader_version':'old'},new_grade={'eval_grader_version':'new'},affected_checkpoints=['hash'],notes='Unit fixture, no real discovery.')
            report = summarize(root/'audit.jsonl')
            self.assertEqual(report['model_observed_root_causes'],0)
            self.assertEqual(report['authored_counterexample_root_causes'],1)
            self.assertEqual(report['pending_reevaluations'],2)

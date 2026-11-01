import json
import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.graders.suite import run_suite


class SuiteTests(unittest.TestCase):
    def test_labelled_failure_output_and_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'final.sqlite',sim_time=100,deadline=10000,principal_id='a')
            world.close()
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,6000],deadline_at=10000)
            (root/'spec.json').write_text(json.dumps(spec))
            cases = [dict(case_id='early_finish',category='incomplete',label_source='hand_authored_unit_fixture',expected_success=False,expected_violations=['no_valid_invited_meeting'],spec='spec.json',final_world='final.sqlite')]
            (root/'cases.json').write_text(json.dumps(cases))
            report = run_suite(root,root/'report')
            self.assertEqual(report['negative_cases'],1)
            self.assertEqual(report['negatives_detected'],1)
            self.assertIsNone(report['legal_false_positive_rate'])
            self.assertEqual(report['label_mismatches'],0)
            self.assertEqual(report['category_counts']['incomplete']['negative_cases'],1)
            self.assertTrue((root/'report'/'cases.jsonl').is_file())

    def test_exported_boundary_deletion_and_retry_cases(self):
        from officegym.graders.cases import generate_cases
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(generate_cases(root/'cases'),42)
            report=run_suite(root/'cases',root/'report')
            self.assertEqual(report['cases'],42)
            self.assertEqual(report['label_mismatches'],0)
            rows=[json.loads(line) for line in (root/'report/cases.jsonl').read_text().splitlines()]
            approval_cases=[row for row in rows if row['case_id'].startswith('approval_')]
            self.assertEqual(report['category_counts']['approval_authorization'],dict(cases=7,legal_cases=3,negative_cases=4,negatives_detected=4,false_positives=0,label_mismatches=0))
            self.assertEqual(sum(value['cases'] for value in report['category_counts'].values()),42)
            self.assertEqual(report['category_counts']['todo_creation_scope'],dict(cases=8,legal_cases=3,negative_cases=5,negatives_detected=5,false_positives=0,label_mismatches=0))
            self.assertEqual(report['category_counts']['meeting_creation_scope'],dict(cases=4,legal_cases=1,negative_cases=3,negatives_detected=3,false_positives=0,label_mismatches=0))
            self.assertEqual(report['category_counts']['invitation_membership'],dict(cases=4,legal_cases=2,negative_cases=2,negatives_detected=2,false_positives=0,label_mismatches=0))
            self.assertEqual(len(approval_cases),7)
            self.assertEqual(sum(row['expected_success'] for row in approval_cases),3)
            self.assertTrue(all(row['category']=='approval_authorization' for row in approval_cases))
            for row in rows:
                if row['case_id'] in ('approval_before_expiry','approval_at_expiry','approval_after_expiry'):
                    trace=(root/'cases'/row['case_id']/'actions.jsonl').read_text()
                    self.assertIn('offline_snapshot_mutation',trace)
                    self.assertIn('expires_at',trace)
                if row['case_id'].startswith('deleted_wrong_send_'):
                    self.assertTrue(row['grade']['outcome_ok'])
                    self.assertIn('operation_evidence_inconsistent',row['grade']['hard_violations'])
                    trace=(root/'cases'/row['case_id']/'actions.jsonl').read_text()
                    self.assertIn('offline_snapshot_mutation',trace)

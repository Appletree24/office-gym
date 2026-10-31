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
            self.assertTrue((root/'report'/'cases.jsonl').is_file())

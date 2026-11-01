import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.operator_inputs import check_operator_inputs


class OperatorInputTests(unittest.TestCase):
    def check(self,teacher='explicit-provider/model-revision',usd=1,hours=1):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps({'model':{'teacher':teacher},'operator_budget':{'max_api_cost_usd':usd,'max_gpu_hours':hours}}))
            return check_operator_inputs(path)

    def test_explicit_choices_do_not_claim_credentials_or_jobs(self):
        report=self.check();self.assertTrue(report['operator_inputs_ready']);self.assertFalse(report['credentials_checked']);self.assertFalse(report['jobs_started'])

    def test_placeholders_and_absent_caps_are_reported(self):
        report=self.check(teacher='SET_BY_OPERATOR',usd=None,hours=None);self.assertEqual(len(report['missing_or_invalid']),3);self.assertFalse(report['operator_inputs_ready'])

    def test_nonfinite_boolean_and_nonpositive_caps_rejected(self):
        for value in (True,0,-1,float('inf'),float('nan'),'unlimited'):
            self.assertFalse(self.check(usd=value)['operator_inputs_ready'])

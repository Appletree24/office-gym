import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.eval.plan import build_plan,freeze_plan
from officegym.eval.calibration import audit_calibration
from officegym.harness.versions import HARNESS_VERSION


class CalibrationTests(unittest.TestCase):
    def test_complete_denominator_bounds_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entry=dict(instance_id='dev1',split='dev',scenario_family_id='f',template_id='D01',depth_level=1,perturbation_level=0,dependency_width=1)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'instances':[entry]}))
            plans=[r for r in build_plan([entry],splits=('dev',)) if r['group'] in ('api_simple','api_full')]
            plan=root/'plan.jsonl';digest=freeze_plan(plans,plan)
            rows=[dict(r,synthetic=False,example_only=False,artifact_path='authored-fixture',model_version='authored-fixture',budget_hash='fixture-budget',harness_version=HARNESS_VERSION,success=r['repeat_id']==0,run_plan_sha256=digest) for r in plans]
            results=root/'rows.jsonl'
            def write(values):results.write_text(''.join(json.dumps(r)+'\n' for r in values))
            def fixture_export(path,planned,source):return next({k:v for k,v in r.items() if k!='run_plan_sha256'} for r in rows if all(r[k]==planned[k] for k in ('group','instance_id','repeat_id')))
            # Mocks isolate accounting; no authored row is admitted as real evidence.
            with patch('officegym.eval.calibration.validate_dataset'),patch('officegym.eval.calibration.result_row',side_effect=fixture_export):
                write(rows);report=audit_calibration(manifest,plan,results,root/'audit.json',min_success_rate=.1,max_success_rate=.9)
                self.assertTrue(report['calibration_usable']);self.assertEqual(report['runs'],6);self.assertFalse(report['dataset_frozen'])
                write(rows[:-1])
                with self.assertRaisesRegex(ValueError,'incomplete'):audit_calibration(manifest,plan,results,root/'bad.json',min_success_rate=.1,max_success_rate=.9)
                rows[0]['synthetic']=True;write(rows)
                with self.assertRaisesRegex(ValueError,'measured'):audit_calibration(manifest,plan,results,root/'bad.json',min_success_rate=.1,max_success_rate=.9)

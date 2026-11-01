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
                unusable=audit_calibration(manifest,plan,results,root/'unusable.json',min_success_rate=.5,max_success_rate=.9)
                self.assertFalse(unusable['calibration_usable']);self.assertTrue(unusable['evidence_verified'])
                write(rows[:-1])
                with self.assertRaisesRegex(ValueError,'incomplete'):audit_calibration(manifest,plan,results,root/'bad.json',min_success_rate=.1,max_success_rate=.9)
                for field in ('model_version','budget_hash'):
                    original=rows[0][field];rows[0][field]='different';write(rows)
                    with self.assertRaisesRegex(ValueError,'shared calibration'):audit_calibration(manifest,plan,results,root/'bad.json',min_success_rate=.1,max_success_rate=.9)
                    rows[0][field]=original
                rows[0]['synthetic']=True;write(rows)
                with self.assertRaisesRegex(ValueError,'measured'):audit_calibration(manifest,plan,results,root/'bad.json',min_success_rate=.1,max_success_rate=.9)


    def test_cli_passes_explicit_bounds_and_returns_unusable_exit(self):
        from officegym.cli import main
        import contextlib
        import io
        args=['officegym','audit-dev-calibration','--manifest','manifest','--plan','plan','--results','results','--out','out','--min-success-rate','0.1','--max-success-rate','0.9']
        with patch('sys.argv',args),patch('officegym.eval.calibration.audit_calibration',return_value={'calibration_usable':False}) as audit,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code,2)
        audit.assert_called_once_with('manifest','plan','results','out',min_success_rate=.1,max_success_rate=.9)


    def test_invalid_bounds_fail_before_any_input_files_are_read(self):
        for lower,upper in ((True,.9),(float('nan'),.9),(.9,.1),(-.1,.9),(.1,1.1)):
            with self.assertRaises(ValueError):audit_calibration('absent','absent','absent','absent',min_success_rate=lower,max_success_rate=upper)

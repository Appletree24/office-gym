"""Mocked evidence tests exercise binding; no fixtures represent model results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.eval.export import frozen_binding
from officegym.eval.frozen_results import audit_frozen_results
from officegym.eval.plan import freeze_plan
from officegym.tasks.freeze import digest


class FrozenResultsTests(unittest.TestCase):
    def test_export_requires_complete_unchanged_batch_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();receipt=root/'freeze.json';receipt.write_text('fixture\n')
            manifest=root/'manifest.json';manifest.write_text('fixture\n')
            selection=root/'selection.json'
            self.assertEqual(frozen_binding(run,manifest),{})
            data=dict(dataset_freeze_path=str(receipt),dataset_freeze_sha256=digest(receipt),dataset_sha256=digest(manifest))
            selection.write_text(json.dumps(data))
            with patch('officegym.tasks.freeze.verify_frozen_dataset',return_value={'manifest_sha256':digest(manifest)}) as verify:
                binding=frozen_binding(run,manifest);self.assertEqual(binding['dataset_freeze_sha256'],digest(receipt));verify.assert_called_once_with(receipt,manifest)
                selection.write_text(json.dumps({'dataset_freeze_sha256':digest(receipt)}))
                with self.assertRaisesRegex(ValueError,'incomplete'):frozen_binding(run,manifest)
                selection.write_text(json.dumps({**data,'dataset_sha256':'wrong'}))
                with self.assertRaisesRegex(ValueError,'dataset differs'):frozen_binding(run,manifest)
                selection.write_text(json.dumps(data));receipt.write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'receipt changed'):frozen_binding(run,manifest)

    def test_original_runtime_reexport_and_plan_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);receipt=root/'freeze.json';receipt.write_text('fixture\n')
            plan=root/'plan.jsonl';planned=dict(instance_id='id1',scenario_family_id='f',template_id='T01',split='id_test',depth_level=1,perturbation_level=0,dependency_width=1,group='api_full',repeat_id=0)
            plan_hash=freeze_plan([planned],plan);run=root/'run';run.mkdir();selection=root/'selection.json';selection.write_text(json.dumps({'plan_sha256':plan_hash}))
            actual=dict(planned,success=True,artifact_path=str(run),dataset_freeze_path=str(receipt),dataset_freeze_sha256=digest(receipt))
            row=dict(actual,run_plan_sha256=plan_hash)
            record=dict(manifest_path=str(root/'manifest.json'),manifest_sha256='fixture')
            with patch('officegym.eval.frozen_results.verify_frozen_dataset',return_value=record),patch('officegym.eval.frozen_results.result_row',return_value=actual):
                self.assertEqual(audit_frozen_results([row],plan,receipt)['runtime_rows_verified'],1)
                for changes,message in (({'success':False},'original runtime'),({'dataset_freeze_sha256':'wrong'},'freeze admission'),({'run_plan_sha256':'wrong'},'frozen run plan')):
                    with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,message):audit_frozen_results([{**row,**changes}],plan,receipt)
                with self.assertRaisesRegex(ValueError,'duplicate final'):audit_frozen_results([row,row],plan,receipt)
                selection.write_text(json.dumps({'plan_sha256':'wrong'}))
                with self.assertRaisesRegex(ValueError,'runtime batch'):audit_frozen_results([row],plan,receipt)

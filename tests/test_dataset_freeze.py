"""Freeze lifecycle tests mock calibration, never claim real model evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.tasks.freeze import freeze_dataset,verify_frozen_dataset
from officegym.graders.versions import EVAL_GRADER_VERSION


class DatasetFreezeTests(unittest.TestCase):
    def fixture(self,root):
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'instances':[]}))
        plan=root/'plan.jsonl';plan.write_text('plan\n');Path(str(plan)+'.sha256').write_text('fixture\n')
        artifacts=root/'runtime';artifacts.mkdir();(artifacts/'terminal.json').write_text('original\n')
        results=root/'results.jsonl';results.write_text(json.dumps({'artifact_path':str(artifacts)})+'\n')
        return manifest,plan,results

    def test_frozen_contents_changed_evidence_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,plan,results=self.fixture(root);out=root/'freeze.json'
            with patch('officegym.tasks.freeze.audit_calibration',return_value={'evidence_verified':True,'calibration_usable':True}) as audit,patch('officegym.tasks.freeze.validate_dataset'):
                record=freeze_dataset(manifest,EVAL_GRADER_VERSION,plan,results,out,min_success_rate=.1,max_success_rate=.9)
                self.assertTrue(record['dataset_frozen']);audit.assert_called_once()
                self.assertEqual(verify_frozen_dataset(out,manifest),record)
                with self.assertRaisesRegex(ValueError,'different dataset'):verify_frozen_dataset(out,root/'other.json')
                original=out.read_bytes()
                with self.assertRaises(FileExistsError):freeze_dataset(manifest,EVAL_GRADER_VERSION,plan,results,out,min_success_rate=.1,max_success_rate=.9)
                self.assertEqual(out.read_bytes(),original)
                with patch('officegym.tasks.freeze.code_inventory',return_value={}):
                    with self.assertRaisesRegex(ValueError,'implementation'):verify_frozen_dataset(out)
                (root/'runtime'/'terminal.json').write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'artifacts changed'):verify_frozen_dataset(out)

    def test_failed_or_changing_audit_creates_no_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,plan,results=self.fixture(root);out=root/'freeze.json'
            for report in ({'evidence_verified':True,'calibration_usable':False},{'evidence_verified':False,'calibration_usable':True}):
                with patch('officegym.tasks.freeze.audit_calibration',return_value=report),self.assertRaisesRegex(ValueError,'usable verified'):
                    freeze_dataset(manifest,EVAL_GRADER_VERSION,plan,results,out,min_success_rate=.1,max_success_rate=.9)
                self.assertFalse(out.exists())
            def changed(*args,**kwargs):
                results.write_text(results.read_text()+'\n')
                return {'evidence_verified':True,'calibration_usable':True}
            with patch('officegym.tasks.freeze.audit_calibration',side_effect=changed),self.assertRaisesRegex(ValueError,'changed during'):
                freeze_dataset(manifest,EVAL_GRADER_VERSION,plan,results,out,min_success_rate=.1,max_success_rate=.9)
            self.assertFalse(out.exists());self.assertFalse(Path(str(out)+'.sha256').exists())
            with self.assertRaisesRegex(ValueError,'current exact'):
                freeze_dataset(manifest,'eval-v1',plan,results,out,min_success_rate=.1,max_success_rate=.9)

    def test_batch_rejects_invalid_freeze_before_any_execution(self):
        from officegym.eval.batch import evaluate_batch
        from unittest.mock import Mock
        execute=Mock()
        with patch('officegym.eval.batch.validate_dataset'),patch('officegym.tasks.freeze.verify_frozen_dataset',side_effect=ValueError('frozen artifacts changed')) as verify:
            with self.assertRaisesRegex(ValueError,'frozen artifacts changed'):
                evaluate_batch('missing','missing','missing','missing',group='api_full',split='id_test',max_runs=1,max_generated_tokens=100,execute=execute,dataset_freeze='receipt')
            verify.assert_called_once();execute.assert_not_called()

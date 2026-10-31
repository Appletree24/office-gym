"""Authored format fixtures, not teacher trajectories or genuine training runs."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.checkpoints import bind_formal_training_source,verify_export_admission,record_export
from officegym.training.evidence import TrainingEvidence
from lineage_fixtures import exported_sft_fixture


class FormalAdmissionTests(unittest.TestCase):
    def test_backend_events_cannot_be_promoted_by_a_passed_equivalence_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'probe_adapter';checkpoint.mkdir()
            (checkpoint/'adapter.safetensors').write_bytes(b'authored backend fixture')
            TrainingEvidence(root).checkpoint(checkpoint,2)
            fake=root/'fabricated_equivalence.json';fake.write_text('{"passed":true,"checks":["fabricated"]}')
            with self.assertRaisesRegex(ValueError,'backend probes cannot become formal exports'):
                record_export(checkpoint,root/'export.json',stage='sft',parent_hash='invented',training_manifest_hash='invented',versions={},equivalence_report=fake,training_events=root/'training_events.jsonl',source_checkpoint=checkpoint,optimizer_step=2)
            self.assertFalse((root/'export.json').exists())

    def test_export_revalidates_data_config_parent_and_equivalence_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,export,record=exported_sft_fixture(root)
            self.assertEqual(verify_export_admission(record),1)
            admitted=record['training_source']['admission']
            paths=[Path(admitted[key]['path']) for key in ('data_manifest','config','data_file')]
            paths.extend([Path(admitted['parent_checkpoint'])/'model.safetensors',Path(record['equivalence_report']),checkpoint/'model.safetensors'])
            for path in paths:
                before=path.read_bytes();path.write_bytes(before+b' changed')
                try:
                    with self.subTest(path=path.name),self.assertRaises((ValueError,json.JSONDecodeError)):
                        verify_export_admission(record)
                finally:path.write_bytes(before)
            self.assertEqual(verify_export_admission(record),1)
            with self.assertRaisesRegex(ValueError,'admission provenance'):
                verify_export_admission(dict(record,export_record_version='checkpoint-export-v2-training-source'))

    def test_stage_parent_data_hash_and_step_budget_bind_to_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,_,record=exported_sft_fixture(root)
            source=record['training_source'];admitted=source['admission']
            args=dict(events_path=source['events_path'],checkpoint=checkpoint,optimizer_step=1,stage='sft',parent_hash=record['parent_hash'],manifest_hash=record['training_manifest_hash'])
            for changes in ({'stage':'grpo'},{'parent_hash':'different'},{'manifest_hash':'different'},{'optimizer_step':0},{'optimizer_step':101}):
                with self.subTest(changes=changes),self.assertRaises(ValueError):bind_formal_training_source(**dict(args,**changes))
            TrainingEvidence(root).event('training_log',2,{'loss':.1})
            self.assertEqual(verify_export_admission(record),1)

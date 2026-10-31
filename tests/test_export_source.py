"""All serialized model files/equivalence claims here are authored test fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.checkpoints import checkpoint_fingerprint,record_export,verify_training_source
from officegym.training.evidence import TrainingEvidence
from lineage_fixtures import admit_fixture


class ExportSourceTests(unittest.TestCase):
    def setup_source(self,root):
        native=root/'native';native.mkdir()
        (native/'model_world_size_1_rank_0.pt').write_bytes(b'not-real-FSDP-shard')
        exported=root/'merged';exported.mkdir()
        (exported/'config.json').write_text('{}')
        (exported/'model.safetensors').write_bytes(b'not-real-merged-weights')
        evidence=TrainingEvidence(root)
        self.admission=admit_fixture(root,evidence)
        source=evidence.checkpoint(native,7)
        report=root/'equivalence.json'
        report.write_text(json.dumps(dict(passed=True,checks=['fixture-not-actual-equivalence'],exported_checkpoint_sha256=checkpoint_fingerprint(exported)['checkpoint_sha256'],source_checkpoint_files_sha256=source['checkpoint_files_sha256'],optimizer_step=7)))
        return native,exported,evidence,report

    def test_source_prefix_survives_append_but_rejects_history_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native,exported,evidence,report=self.setup_source(root)
            record=record_export(exported,root/'export.json',stage='sft',parent_hash=self.admission['parent_checkpoint_sha256'],training_manifest_hash=self.admission['data_manifest']['sha256'],versions={},equivalence_report=report,training_events=root/'training_events.jsonl',source_checkpoint=native,optimizer_step=7)
            evidence.event('training_log',8,{'loss':1})
            self.assertEqual(verify_training_source(record['training_source']),7)
            events=root/'training_events.jsonl'
            events.write_text(events.read_text().replace('"optimizer_step":7','"optimizer_step":70'))
            with self.assertRaises(ValueError):
                verify_training_source(record['training_source'])

    def test_equivalence_must_bind_actual_exported_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native,exported,evidence,report=self.setup_source(root)
            (exported/'model.safetensors').write_bytes(b'changed-export')
            with self.assertRaisesRegex(ValueError,'does not bind exported'):
                record_export(exported,root/'export.json',stage='sft',parent_hash=self.admission['parent_checkpoint_sha256'],training_manifest_hash=self.admission['data_manifest']['sha256'],versions={},equivalence_report=report,training_events=root/'training_events.jsonl',source_checkpoint=native,optimizer_step=7)

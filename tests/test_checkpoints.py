import tempfile
import unittest
from pathlib import Path
from officegym.training.checkpoints import checkpoint_fingerprint,validate_branches


class CheckpointTests(unittest.TestCase):
    def test_weight_mutation_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'unit fixture, not real weights')
            original = checkpoint_fingerprint(root)
            (root/'model.safetensors').write_bytes(b'modified fixture')
            changed = checkpoint_fingerprint(root)
            self.assertNotEqual(original['checkpoint_sha256'],changed['checkpoint_sha256'])

    def test_rs_sft_cannot_start_after_grpo(self):
        from lineage_fixtures import exported_sft_fixture
        with tempfile.TemporaryDirectory() as directory:
            _,_,sft=exported_sft_fixture(Path(directory))
            self.assertEqual(validate_branches(sft,sft,sft,sft),sft['checkpoint_sha256'])
            with self.assertRaises(ValueError):
                validate_branches(sft,sft,sft,dict(sft,stage='grpo'))
            with self.assertRaises(ValueError):
                validate_branches({'stage':'sft','checkpoint_sha256':'unverified'},sft,sft,sft)

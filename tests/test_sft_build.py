import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.build import build_dataset


class BuildTests(unittest.TestCase):
    def test_rejections_recorded_without_tokenizer_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'run'
            run.mkdir()
            (run/'result.json').write_text(json.dumps({'split':'id_test'}))
            result = build_dataset([run],root/'data',lambda *args:self.fail('test data tokenized'),tokenizer_manifest={})
            self.assertEqual(result['accepted'],[])
            self.assertEqual(result['rejection_counts'],{'non_training_split':1})
            self.assertEqual(result['manual_alignment_review_status'],'pending')
            self.assertTrue((root/'data'/'rejections.jsonl').is_file())

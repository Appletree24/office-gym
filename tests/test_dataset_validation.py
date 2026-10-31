import json
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.validate import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_empty_and_missing_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'manifest.json'
            path.write_text(json.dumps({'instances':[]}))
            with self.assertRaisesRegex(ValueError,'empty dataset'):
                validate_dataset(path)
            path.write_text(json.dumps({'instances':[{'instance_id':'one','split':'train','exact_fingerprint':'hash','path':'one','files_sha256':{}}]}))
            with self.assertRaisesRegex(ValueError,'missing artifact'):
                validate_dataset(path)

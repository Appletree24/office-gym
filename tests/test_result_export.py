import json
import tempfile
import unittest
from pathlib import Path
from officegym.eval.export import result_row


class ExportTests(unittest.TestCase):
    def test_software_probe_never_exported_as_model_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'result.json').write_text(json.dumps({'policy_kind':'software_fixture'}))
            with self.assertRaisesRegex(ValueError,'software runs'):
                result_row(root,{},root/'manifest.json')

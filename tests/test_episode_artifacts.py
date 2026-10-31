import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from officegym.eval.artifacts import write_json


class EpisodeArtifactTests(unittest.TestCase):
    def test_failed_publish_keeps_previous_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'result.json'
            write_json(target,{'complete':True,'attempt':1})
            with patch('officegym.eval.artifacts.os.replace',side_effect=OSError('fixture interrupted publish')):
                with self.assertRaises(OSError):
                    write_json(target,{'complete':True,'attempt':2})
            self.assertEqual(json.loads(target.read_text()),{'complete':True,'attempt':1})
            self.assertEqual([path.name for path in target.parent.iterdir()],['result.json'])
            with self.assertRaises(ValueError):
                write_json(target,{'bad':float('nan')})
            self.assertEqual(json.loads(target.read_text())['attempt'],1)

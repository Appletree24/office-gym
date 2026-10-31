import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.teacher import extract_trajectory
from officegym.training.data import RejectedTrajectory


class TeacherExtractTests(unittest.TestCase):
    def test_nontraining_and_scripted_runs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'result.json').write_text(json.dumps({'split':'id_test'}))
            with self.assertRaisesRegex(RejectedTrajectory,'non_training_split'):
                extract_trajectory(root)
            (root/'result.json').write_text(json.dumps({'split':'train','policy_kind':'software_fixture'}))
            with self.assertRaisesRegex(RejectedTrajectory,'not_a_model'):
                extract_trajectory(root)

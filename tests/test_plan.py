import tempfile
import unittest
from pathlib import Path
from officegym.eval.plan import build_plan,freeze_plan,missing_runs


class PlanTests(unittest.TestCase):
    def test_complete_matrix_and_missing(self):
        instances = [dict(instance_id='i',split='id_test',scenario_family_id='family',template_id='T01',depth_level=1,perturbation_level=0,dependency_width=1)]
        rows = build_plan(instances,splits=('id_test',))
        self.assertEqual(len(rows),15)
        self.assertEqual({row['repeat_id'] for row in rows},{0,1,2})
        self.assertEqual(missing_runs(rows,rows[:-1]),rows[-1:])
        with self.assertRaises(ValueError):
            missing_runs(rows,rows+[rows[0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.jsonl'
            fingerprint = freeze_plan(rows,path)
            self.assertEqual(Path(str(path)+'.sha256').read_text().strip(),fingerprint)
            with self.assertRaises(FileExistsError):
                freeze_plan(rows,path)

    def test_empty_split_rejected(self):
        with self.assertRaises(ValueError):
            build_plan([],splits=('id_test',))

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tasks.validate import validate_dataset
from officegym.training.environment import RolloutEnvironment


class PartitionTests(unittest.TestCase):
    def test_relabelled_unseen_task_rejected_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'data',templates=['U01'],count_per_template=1,split='unseen_test')
            entry=manifest['instances'][0]
            source=root/'data'/entry['path']
            task_path=source/'task.json'
            task=json.loads(task_path.read_text())
            task.update(split='train',seed=100000)
            task_path.write_text(json.dumps(task))
            entry['split']='train'
            entry['files_sha256']['task.json']=hashlib.sha256(task_path.read_bytes()).hexdigest()
            (root/'data'/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'template is not permitted'):
                validate_dataset(root/'data'/'manifest.json')
            with self.assertRaisesRegex(ValueError,'template is not permitted'):
                RolloutEnvironment(source,actor_version='fixture')

    def test_relabelled_probe_seed_and_disabled_holdout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'data',templates=['T01'],count_per_template=1)
            entry=manifest['instances'][0]
            task_path=root/'data'/entry['path']/'task.json'
            task=json.loads(task_path.read_text())
            task['split']='train'
            task_path.write_text(json.dumps(task))
            entry['split']='train'
            entry['files_sha256']['task.json']=hashlib.sha256(task_path.read_bytes()).hexdigest()
            path=root/'data'/'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'seed is outside'):
                validate_dataset(path)
            manifest['heldout_signatures']=[]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'fixed ordered holdout'):
                validate_dataset(path)

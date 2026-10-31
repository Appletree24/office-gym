import json
import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tasks.artifacts import write_instance
from officegym.tasks.validate import validate_dataset


class ArtifactTests(unittest.TestCase):
    def test_write_reference_validate_and_detect_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'seed.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            public = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000)
            task = dict(instance_id='test_task',split='probe',template_id='T01',scenario_family_id='f',depth_level=1,perturbation_level=0,public_task=public,dependency_dag={'nodes':['plan','invite'],'edges':[['plan','invite']]})
            entry = write_instance(root/'tasks',task,world,{},[])
            world.close()
            manifest = root/'tasks'/'manifest.json'
            manifest.write_text(json.dumps({'instances':[entry]}))
            self.assertEqual(validate_dataset(manifest)['instances'],1)
            entry['family_fingerprint'] = 'forged'
            manifest.write_text(json.dumps({'instances':[entry]}))
            with self.assertRaisesRegex(ValueError,'semantic fingerprint mismatch'):
                validate_dataset(manifest)
            entry['family_fingerprint'] = entry['exact_fingerprint']
            manifest.write_text(json.dumps({'instances':[entry]}))
            (root/'tasks'/'test_task'/'task.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                validate_dataset(manifest)

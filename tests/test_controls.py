import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tasks.validate import validate_dataset
from officegym.tasks.controls import PROFILE


class ControlledNuisanceTests(unittest.TestCase):
    def test_five_cells_share_actual_nuisance_and_timing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);documents=[];background=[]
            for depth,perturbation in ((1,1),(2,1),(3,1),(2,0),(2,2)):
                path=root/('d'+str(depth)+'p'+str(perturbation))
                manifest=generate(path,templates=['T04'],count_per_template=1,depth_level=depth,perturbation_level=perturbation)
                entry=manifest['instances'][0];source=path/entry['path']
                task=json.loads((source/'task.json').read_text())
                self.assertEqual(task['control_profile'],PROFILE)
                self.assertEqual(task['dependency_width'],1)
                self.assertEqual(task['public_task'].get('proposal_bound_availability',False),depth>=2)
                self.assertEqual(validate_dataset(path/'manifest.json')['instances'],1)
                with sqlite3.connect(source/'initial_world.sqlite') as db:
                    documents.append(db.execute('SELECT content FROM documents').fetchone()[0])
                    background.append(db.execute('SELECT body FROM messages ORDER BY id').fetchall())
                self.assertEqual(len(documents[-1]),1024)
                self.assertEqual(len(background[-1]),3)
                if depth==1:
                    self.assertNotIn('required_document_refs',task['public_task'])
            self.assertEqual(len(set(documents)),1)
            self.assertTrue(all(messages==background[0] for messages in background))

    def test_rehashed_noise_change_breaks_control_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T04'],count_per_template=1,depth_level=2,perturbation_level=1)
            entry=manifest['instances'][0];initial=root/'tasks'/entry['path']/'initial_world.sqlite'
            with sqlite3.connect(initial) as db:
                db.execute("UPDATE messages SET body='short' WHERE id='background_0'")
            entry['files_sha256']['initial_world.sqlite']=hashlib.sha256(initial.read_bytes()).hexdigest()
            (root/'tasks'/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'initial noise/document/busy state'):
                validate_dataset(root/'tasks'/'manifest.json')

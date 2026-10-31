import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate,TRAIN_TEMPLATES
from officegym.tasks.validate import validate_dataset


class GeneratorTests(unittest.TestCase):
    def test_all_training_templates_reference_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tasks'
            manifest = generate(root,count_per_template=1)
            self.assertEqual({entry['template_id'] for entry in manifest['instances']},set(TRAIN_TEMPLATES))
            self.assertEqual(len(TRAIN_TEMPLATES),8)
            for entry in manifest['instances']:
                initial = sqlite3.connect(root/entry['path']/'initial_world.sqlite')
                try:
                    calls = json.loads(initial.execute("SELECT value_json FROM meta WHERE key='tool_calls'").fetchone()[0])
                    self.assertEqual(calls,0)
                    self.assertEqual(initial.execute('SELECT count(*) FROM operations').fetchone()[0],0)
                finally:
                    initial.close()
            report = validate_dataset(root/'manifest.json')
            self.assertEqual(report['instances'],8)
            conflict = next(entry for entry in manifest['instances'] if entry['template_id']=='T03')
            db = sqlite3.connect(root/conflict['path']/'reference_final.sqlite')
            try:
                rows = db.execute('SELECT title,start_at,end_at FROM meetings ORDER BY start_at').fetchall()
                self.assertEqual(rows[0][0],'Existing commitment')
                self.assertGreaterEqual(rows[1][1],rows[0][2])
            finally:
                db.close()

    def test_split_and_seed_leakage_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tasks'
            for options in ({'templates':['D01'],'split':'train'}, {'templates':['T01'],'split':'dev'}, {'templates':['T01'],'split':'train','seed':200000}, {'templates':['T01'],'split':'train','seed':199999,'count_per_template':2}):
                with self.assertRaises(ValueError):
                    generate(root,**options)
                self.assertFalse(root.exists())

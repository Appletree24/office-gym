import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tasks.validate import validate_dataset


class EventContractTests(unittest.TestCase):
    def test_rehashed_perturbation_label_cannot_hide_scheduled_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T06'],count_per_template=1)
            entry=manifest['instances'][0]
            task_path=root/'tasks'/entry['path']/'task.json'
            task=json.loads(task_path.read_text())
            task['perturbation_level']=0
            entry['perturbation_level']=0
            task_path.write_text(json.dumps(task))
            entry['files_sha256']['task.json']=hashlib.sha256(task_path.read_bytes()).hexdigest()
            (root/'tasks'/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'perturbation label differs'):
                validate_dataset(root/'tasks'/'manifest.json')

    def test_declared_schedule_and_rule_must_match_initial_database(self):
        for change in ('schedule','rule','public_task'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                manifest=generate(root/'tasks',templates=['T06'],count_per_template=1)
                entry=manifest['instances'][0]
                initial=root/'tasks'/entry['path']/'initial_world.sqlite'
                with sqlite3.connect(initial) as db:
                    if change=='schedule':
                        db.execute('UPDATE scheduled_events SET priority=99')
                    elif change=='rule':
                        db.execute("UPDATE meta SET value_json='{}' WHERE key='approval_rules'")
                    else:
                        db.execute("UPDATE meta SET value_json='{}' WHERE key='public_task'")
                entry['files_sha256']['initial_world.sqlite']=hashlib.sha256(initial.read_bytes()).hexdigest()
                (root/'tasks'/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'differ'):
                    validate_dataset(root/'tasks'/'manifest.json')

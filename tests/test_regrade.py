import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import canonical
from officegym.tasks.generate import generate
from officegym.tasks.regrade import regrade,digest
from officegym.graders.versions import EVAL_GRADER_VERSION


class RegradeTests(unittest.TestCase):
    def test_regrade_preserves_inputs_and_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'source',templates=['T08'],count_per_template=1)
            source=root/'source/manifest.json';original=digest(source)
            with self.assertRaisesRegex(ValueError,'outside source'):regrade(source,root/'source/nested')
            self.assertFalse((root/'source/nested').exists())
            report=regrade(source,root/'rescored',max_wall_seconds=10)
            self.assertEqual(report['reference_passed'],1);self.assertEqual(digest(source),original)
            old=json.loads(source.read_text());new=json.loads((root/'rescored/manifest.json').read_text())
            for entry,updated in zip(old['instances'],new['instances']):
                self.assertEqual(entry['reference_final_hash'],updated['reference_final_hash'])
                for name in ('task.json','initial_world.sqlite','private_spec.json','event_rules.json','reference_trace.jsonl'):
                    self.assertEqual(digest(source.parent/entry['path']/name),digest(root/'rescored'/updated['path']/name))
                grade=json.loads((root/'rescored'/updated['path']/'reference_grade.json').read_text())
                self.assertEqual(grade['eval_grader_version'],EVAL_GRADER_VERSION)
            with self.assertRaises(FileExistsError):regrade(source,root/'rescored')

    def test_frozen_or_corrupted_sources_are_not_silently_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'source',templates=['T08'],count_per_template=1)
            path=root/'source/manifest.json';manifest=json.loads(path.read_text())
            manifest['status']='frozen';path.write_text(canonical(manifest)+'\n')
            with self.assertRaisesRegex(ValueError,'unfrozen'):regrade(path,root/'frozen')
            self.assertFalse((root/'frozen').exists())
            manifest['status']='development_dataset_not_frozen';path.write_text(canonical(manifest)+'\n')
            target=path.parent/manifest['instances'][0]['path']/'task.json';target.write_text(target.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):regrade(path,root/'bad')

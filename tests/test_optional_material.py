import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import canonical
from officegym.tasks.generate import generate
from officegym.tasks.validate import validate_dataset
from officegym.tasks.regrade import regrade
from officegym.tasks.integrity import dependency_depth,dependency_width
from officegym.graders.optional_read_probe import run


class OptionalMaterialTests(unittest.TestCase):
    def test_public_version_binding_can_complete_without_read_and_is_not_an_intrinsic_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T05'],count_per_template=1)
            entry=manifest['instances'][0];task=json.loads((root/'tasks'/entry['path']/'task.json').read_text())
            self.assertNotIn('read_material',task['dependency_dag']['nodes']);self.assertEqual(task['optional_observation_tools'],['documents.read'])
            report=run(root/'tasks/manifest.json',root/'probe',max_instances=1,max_wall_seconds=10)
            self.assertEqual(report['declared_read_nodes'],0);self.assertTrue(report['all_read_omissions_legal']);self.assertGreater(report['skipped_reads'],0);self.assertEqual(report['actual_material_reads'],0);self.assertTrue(report['all_calls_use_real_dispatcher'])

    def test_legacy_observation_annotation_requires_explicit_migration_without_changing_business_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T05'],count_per_template=1);entry=manifest['instances'][0];path=root/'tasks'/entry['path']/'task.json';task=json.loads(path.read_text())
            task['dependency_dag']=dict(nodes=['plan','read_material','confirm','invite'],edges=[['plan','confirm'],['confirm','invite'],['read_material','plan']]);task['dependency_path_length']=dependency_depth(**task['dependency_dag']);task['dependency_width']=dependency_width(**task['dependency_dag']);task.pop('dependency_annotation_version');task.pop('optional_observation_tools')
            path.write_text(canonical(task)+'\n');entry['files_sha256']['task.json']=hashlib.sha256(path.read_bytes()).hexdigest();entry['dependency_width']=task['dependency_width'];manifest['task_generator_version']='generator-v3-controlled-width';(root/'tasks/manifest.json').write_text(canonical(manifest)+'\n');before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'optional material'):validate_dataset(root/'tasks/manifest.json')
            result=regrade(root/'tasks/manifest.json',root/'reannotated',repair_optional_material=True,max_wall_seconds=10)
            self.assertEqual(result['task_metadata_reannotated'],1);self.assertFalse(result['task_inputs_unchanged']);self.assertTrue(result['public_business_inputs_unchanged']);self.assertEqual(path.read_bytes(),before)
            updated=json.loads((root/'reannotated'/entry['path']/'task.json').read_text());self.assertEqual(updated['public_task'],task['public_task']);self.assertEqual(updated['dependency_path_length'],3)
            self.assertEqual(validate_dataset(root/'reannotated/manifest.json')['instances'],1)

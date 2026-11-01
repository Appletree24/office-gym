import tempfile
from pathlib import Path
import unittest
from officegym.tasks.generate import generate
from officegym.graders.dependency_probe import run_dependency_probes,run_dataset_probes


class DependencyPoolTests(unittest.TestCase):
    def test_template_specific_stages_and_owner_chain_targeted_without_deadline_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for template,split,expected in [('T08','train',2),('U04','unseen_test',4)]:
                manifest=generate(root/template,templates=[template],count_per_template=1,split=split)
                summary=run_dependency_probes(root/template/manifest['instances'][0]['path'],root/(template+'_probe'))
                self.assertEqual(summary['cases'],expected);self.assertTrue(summary['all_stages_required'])
                if template=='U04':self.assertTrue({'owner_reassignment','owner_followup'}<=set(summary['selected_stages']))

    def test_pool_cap_cannot_silently_change_full_coverage_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T08'],count_per_template=2)
            with self.assertRaisesRegex(ValueError,'complete D3'):run_dataset_probes(root/'tasks/manifest.json',root/'small',max_instances=1,max_wall_seconds=10)
            self.assertFalse((root/'small').exists())
            result=run_dataset_probes(root/'tasks/manifest.json',root/'all',max_instances=2,max_wall_seconds=10)
            self.assertEqual(result['instances'],2);self.assertEqual(result['cases'],4);self.assertTrue(result['all_target_constraints'])

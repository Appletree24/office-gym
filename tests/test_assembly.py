import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from officegym.tasks.assemble import assemble
from officegym.tasks.composition import report


class AssemblyTests(unittest.TestCase):
    def test_id_conditions_are_actually_sampled_in_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            counts=dict(train=40,dev=2,id_test=5,combo_test=1,unseen_test=4,total=52)
            config=root/'config.json';config.write_text(json.dumps({'counts':counts}))
            self.assertEqual(assemble(config,root/'dataset')['instances'],52)
            manifest=json.loads((root/'dataset'/'manifest.json').read_text())
            train=[entry for entry in manifest['instances'] if entry['split']=='train']
            self.assertEqual(set(Counter(entry['template_id'] for entry in train).values()),{5})
            grid={(entry['depth_level'],entry['perturbation_level']) for entry in train if entry['template_id']=='T04'}
            id_grid={(entry['depth_level'],entry['perturbation_level']) for entry in manifest['instances'] if entry['split']=='id_test'}
            self.assertEqual(grid,id_grid)
            self.assertEqual(len(grid),5)
            composition=report(root/'dataset'/'manifest.json',root/'composition.json')
            self.assertIn(dict(split='train',profile='controlled-nuisance-v1',count=5),composition['control_profiles'])

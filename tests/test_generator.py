import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate,TRAIN_TEMPLATES,SUPPORTED
from officegym.tasks.replay import reference_manifest
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
            signatures = {}
            for entry in manifest['instances']:
                task = json.loads((root/entry['path']/'task.json').read_text())
                signatures[entry['template_id']] = task['event_signature']
            self.assertEqual(signatures['T05'],['DocumentVersionPublished'])
            self.assertEqual(signatures['T07'],['AttendeeChanged'])
            self.assertFalse(any('AttendeeChanged' in signature and 'DocumentVersionPublished' in signature for signature in signatures.values()))
            material = next(entry for entry in manifest['instances'] if entry['template_id']=='T05')
            final = sqlite3.connect(root/material['path']/'reference_final.sqlite')
            try:
                self.assertEqual(final.execute("SELECT max(version) FROM documents WHERE document_id='agenda'").fetchone()[0],2)
                refs = final.execute('SELECT document_refs_json FROM meetings').fetchone()[0]
                self.assertEqual(json.loads(refs),[dict(document_id='agenda',version=2)])
            finally:
                final.close()
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

    def test_all_fourteen_templates_replay_and_partition_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=SUPPORTED,count_per_template=1)
            self.assertEqual(len(manifest['instances']),14)
            self.assertEqual(validate_dataset(root/'tasks'/'manifest.json')['templates'],14)
            replay = reference_manifest(root/'tasks'/'manifest.json',root/'replay')
            self.assertEqual(replay['reference_passed'],14)
            self.assertEqual(replay['replay_matched'],14)
            for template in ('U01','U02','U03','U04'):
                with self.assertRaises(ValueError):
                    generate(root/'invalid',templates=[template],split='train')
                self.assertFalse((root/'invalid').exists())

    def test_heldout_combo_cannot_be_generated_in_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tasks'
            with self.assertRaisesRegex(ValueError,'held-out combinations'):
                generate(root,templates=['T07'],split='train',heldout_combo=True)
            self.assertFalse(root.exists())

    def test_controlled_five_cell_scan_is_solvable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for depth,perturbation in ((1,1),(2,1),(3,1),(2,0),(2,2)):
                manifest = generate(root/('d'+str(depth)+'p'+str(perturbation)),templates=['T04'],count_per_template=1,depth_level=depth,perturbation_level=perturbation)
                entry = manifest['instances'][0]
                self.assertEqual(entry['depth_level'],depth)
                self.assertEqual(entry['perturbation_level'],perturbation)
                self.assertEqual(entry['scenario_family_id'],'family_900000')
                self.assertTrue(json.loads((root/('d'+str(depth)+'p'+str(perturbation))/entry['path']/'reference_grade.json').read_text())['full_success'])

    def test_combo_atoms_are_seen_but_combination_is_held_out(self):
        from officegym.tasks.composition import report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = generate(root/'train',count_per_template=1,split='train')
            combo = generate(root/'combo',templates=['T07'],count_per_template=1,split='combo_test',heldout_combo=True)
            entries = [{**entry,'path':prefix+'/'+entry['path']} for prefix,manifest in [('train',train),('combo',combo)] for entry in manifest['instances']]
            manifest = dict(instances=entries,heldout_signatures=train['heldout_signatures'])
            (root/'manifest.json').write_text(json.dumps(manifest))
            coverage = report(root/'manifest.json',root/'composition.json')['combo_atom_coverage']
            self.assertTrue(coverage['all_combo_atoms_seen_in_training'])
            self.assertEqual(coverage['missing_training_event_kinds'],[])
            manifest['instances'] = [entry for entry in entries if entry['template_id']!='T05']
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'atoms absent from training'):
                validate_dataset(root/'manifest.json')

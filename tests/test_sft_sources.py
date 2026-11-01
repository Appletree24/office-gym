"""Teacher source bindings tested with explicitly authored parser fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import canonical
from officegym.training.sft_sources import verify_sources,source_hashes
from officegym.training.build import build_dataset
from review_fixtures import reviewed_fixture,bind_authored_sources
from test_mask_review import fixture


class SFTSourceTests(unittest.TestCase):
    def test_complete_inventory_and_each_retained_file_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data';reviewed_fixture(data)
            manifest=json.loads((data/'manifest.json').read_text())
            self.assertEqual(verify_sources(data,manifest)['source_files'],40)
            for name in ('result.json','trajectory.jsonl','policy_isolation.json','task.json','private_spec.json','event_rules.json','initial_world.sqlite','world.sqlite'):
                path=Path(manifest['accepted'][0]['source_run'])/name;before=path.read_bytes();path.write_bytes(before+b' ')
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'evidence changed'):verify_sources(data,manifest)
                path.write_bytes(before)
            del manifest['accepted'][0]['source_files_sha256']
            with self.assertRaisesRegex(ValueError,'complete hash-bound'):verify_sources(data,manifest)

    def test_rehashing_changed_evidence_does_not_hide_conversation_or_grade_drift(self):
        for fault in ('trace','grade','world','input','duplicate_path','sample_source','token_total'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                data=Path(directory)/'data';reviewed_fixture(data);manifest=json.loads((data/'manifest.json').read_text());entry=manifest['accepted'][0];source=Path(entry['source_run'])
                if fault=='trace':
                    path=source/'trajectory.jsonl';path.write_text(path.read_text().replace('authored_software_fixture','changed_authored_fixture'));entry['source_files_sha256']=source_hashes(source)
                elif fault=='grade':
                    path=source/'result.json';result=json.loads(path.read_text());result['grade']['full_success']=False;path.write_text(canonical(result)+'\n');entry['source_files_sha256']=source_hashes(source)
                elif fault=='world':
                    import sqlite3
                    with sqlite3.connect(source/'world.sqlite') as db:db.execute("UPDATE meta SET value_json='false' WHERE key='finished'")
                    entry['source_files_sha256']=source_hashes(source)
                elif fault=='input':
                    import hashlib
                    path=source/'task.json';task=json.loads(path.read_text());task['public_task']['title']='Changed fixture';path.write_text(canonical(task)+'\n')
                    result_path=source/'result.json';result=json.loads(result_path.read_text());result['versions']['instance_artifact_sha256']['task.json']=hashlib.sha256(path.read_bytes()).hexdigest();result_path.write_text(canonical(result)+'\n')
                    entry['source_files_sha256']=source_hashes(source)
                elif fault=='duplicate_path':manifest['accepted'][1]['source_run']=entry['source_run']
                elif fault=='sample_source':
                    path=data/'samples.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]['source_run']='/other/fixture';path.write_text(''.join(canonical(row)+'\n' for row in rows))
                else:manifest['training_tokens']+=1
                with self.assertRaises(ValueError):verify_sources(data,manifest)

    def test_builder_writes_current_source_inventory_and_verifier_accepts_its_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'fixtures';fixture(data)
            manifest=json.loads((data/'manifest.json').read_text());tokenizer=bind_authored_sources(data,manifest)
            runs=[entry['source_run'] for entry in manifest['accepted']]
            built=build_dataset(runs,root/'built',tokenizer,tokenizer_manifest={'fixture':True})
            self.assertEqual(len(built['accepted']),5)
            self.assertEqual(verify_sources(root/'built',built)['training_tokens'],15)
            self.assertEqual(built['manual_alignment_review_status'],'pending')
            self.assertTrue(all(set(row['source_files_sha256'])=={'result.json','trajectory.jsonl','policy_isolation.json','task.json','private_spec.json','event_rules.json','initial_world.sqlite','world.sqlite'} for row in built['accepted']))

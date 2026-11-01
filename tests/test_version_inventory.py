import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.export_versions import export


class VersionInventoryTests(unittest.TestCase):
    def test_configs_locks_and_launchers_are_hashed_without_reading_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources={'src/officegym/core.py':'x=1\n','schema/world.sql':'SELECT 1;','contracts/tools.json':'{}','configs/train.json':'{"max_steps":5}','configs/native/resolved.yaml':'engine: fsdp2\n','requirements-training.lock':'torch==1\n','scripts/train_grpo.sh':'#!/bin/sh\nexit 0\n','pyproject.toml':'[project]\nname="fixture"\n','artifacts/model/weights.py':'excluded','configs/token.secret':'excluded'}
            for name,content in sources.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
            first=export(root);second=export(root)
            self.assertEqual(first['source_manifest_sha256'],second['source_manifest_sha256'])
            expected=set(sources)-{'artifacts/model/weights.py','configs/token.secret'}
            self.assertEqual(set(first['file_sha256']),expected)
            for name in expected:self.assertEqual(first['file_sha256'][name],hashlib.sha256((root/name).read_bytes()).hexdigest())
            self.assertEqual(first['inventory_groups']['configuration'],['configs/native/resolved.yaml','configs/train.json'])
            self.assertEqual(first['inventory_groups']['dependency_locks'],['requirements-training.lock'])
            for name in ('configs/train.json','requirements-training.lock','scripts/train_grpo.sh'):
                (root/name).write_text((root/name).read_text()+'\n')
                changed=export(root);self.assertNotEqual(first['source_manifest_sha256'],changed['source_manifest_sha256']);self.assertNotEqual(first['file_sha256'][name],changed['file_sha256'][name])

    def test_empty_inventory_is_explicit_and_does_not_claim_a_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result=export(directory)
            self.assertEqual(result['file_sha256'],{});self.assertIsNone(result['git_commit']);self.assertIsNone(result['git_dirty']);self.assertIn('not a resolved runtime',result['scope'])

    def test_cli_preserves_existing_version_evidence(self):
        import subprocess
        import sys
        import json
        script=Path(__file__).resolve().parents[1]/'scripts/export_versions.py'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);destination=root/'versions.json'
            command=[sys.executable,str(script),'--root',str(root),'--out',str(destination)]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            original=destination.read_bytes();self.assertEqual(json.loads(original)['kind'],'software_version_manifest_not_experiment')
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0);self.assertIn('FileExistsError',second.stderr);self.assertEqual(destination.read_bytes(),original)

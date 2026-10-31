import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.training.registry import RolloutRegistry


class RegistryTests(unittest.TestCase):
    def test_tools_share_world_and_release_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source = root/'tasks'/manifest['instances'][0]['path']
            registry = RolloutRegistry()
            registry.acquire('trajectory','board.patch',source,'sft0')
            registry.acquire('trajectory','board.read',source,'sft0')
            registry.call('trajectory','board.patch',{'expected_version':1,'patch':{'goal':'shared'}})
            self.assertEqual(registry.call('trajectory','board.read',{})['data']['version'],2)
            with self.assertRaises(ValueError):
                registry.acquire('trajectory','mail.send',source,'changed_actor')
            registry.release('trajectory','board.patch')
            self.assertIn('trajectory',registry.entries)
            registry.release('trajectory','board.read')
            self.assertEqual(registry.entries,{})

    def test_episode_lease_survives_per_call_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source = root/'tasks'/manifest['instances'][0]['path']
            registry = RolloutRegistry()
            registry.acquire('t','__episode__',source,'sft0')
            registry.acquire('t','board.patch',source,'sft0')
            registry.call('t','board.patch',{'expected_version':1,'patch':{'goal':'retained'}})
            registry.release('t','board.patch')
            registry.acquire('t','board.read',source,'sft0')
            self.assertEqual(registry.call('t','board.read',{})['data']['version'],2)
            registry.release('t','board.read')
            registry.release('t','__episode__')
            self.assertEqual(registry.entries,{})

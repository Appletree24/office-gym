import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World


class WorldTests(unittest.TestCase):
    def test_replay_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('a.sqlite', 'b.sqlite')]
            hashes = []
            for path in paths:
                world = World.create(path, sim_time=100, deadline=1000, principal_id='agent')
                hashes.append(world.state_hash())
                world.close()
            self.assertEqual(*hashes)
            world = World(paths[0])
            self.assertEqual(world.state_hash(), hashes[0])
            world.close()
            with self.assertRaises(FileExistsError):
                World.create(paths[0], sim_time=100, deadline=1000, principal_id='agent')

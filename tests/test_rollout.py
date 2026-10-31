import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.env.rollout import clone_world,rollout_world
from officegym.tools.dispatcher import Dispatcher


class RolloutTests(unittest.TestCase):
    def test_wal_snapshot_isolated_and_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'initial.sqlite'
            initial = World.create(source,sim_time=100,deadline=10000,principal_id='a')
            with initial.db:
                initial.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            initial_hash = initial.state_hash()
            with rollout_world(source,work_dir=root) as first, rollout_world(source,work_dir=root) as second:
                self.assertEqual(first.state_hash(),initial_hash)
                self.assertEqual(second.state_hash(),initial_hash)
                first_path = Path(first.db.execute('PRAGMA database_list').fetchone()[2])
                result = Dispatcher(first).call('board.patch',{'expected_version':1,'patch':{'goal':'changed'}},'write')
                self.assertTrue(result['ok'],result)
                self.assertNotEqual(first.state_hash(),second.state_hash())
                self.assertEqual(second.state_hash(),initial.state_hash())
            self.assertFalse(first_path.exists())
            initial.close()

    def test_existing_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=1000,principal_id='a')
            original = world.state_hash()
            with self.assertRaises(FileExistsError):
                clone_world(path,path)
            self.assertEqual(world.state_hash(),original)
            world.close()

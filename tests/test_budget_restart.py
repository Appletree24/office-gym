import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class RestartBudgetTests(unittest.TestCase):
    def test_cannot_extend_budget_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world,max_calls=1)
            self.assertTrue(client.call('board.read',{})['ok'])
            world.close()
            world = World(path)
            with self.assertRaises(ValueError):
                Dispatcher(world,max_calls=200)
            client = Dispatcher(world)
            self.assertEqual(client.max_calls,1)
            self.assertEqual(client.call('board.read',{})['error']['code'],'BUDGET_EXHAUSTED')
            self.assertEqual(client.meta('tool_calls'),1)
            world.close()

    def test_invalid_initial_clock_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            with self.assertRaises(ValueError):
                World.create(path,sim_time=True,deadline=1000,principal_id='a')
            self.assertFalse(path.exists())

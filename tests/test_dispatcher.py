import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class DispatcherTests(unittest.TestCase):
    def test_receipt_survives_reopen_and_conflict_costs_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'world.sqlite'
            world = World.create(path, sim_time=100, deadline=1000, principal_id='agent')
            with world.db:
                world.db.execute("INSERT INTO people VALUES ('agent','Agent','a@x','UTC','[]')")
            client = Dispatcher(world)
            args = {'expected_version': 1, 'patch': {'goal': 'schedule'}}
            first = client.call('board.patch', args, 'op1')
            self.assertTrue(first['ok'])
            world.close()
            world = World(path)
            client = Dispatcher(world)
            replay = client.call('board.patch', args, 'op1')
            self.assertEqual(first['data'], replay['data'])
            self.assertEqual(replay['meta']['sim_time'], 160)
            conflict = client.call('board.patch', dict(args, expected_version=2), 'op1')
            self.assertEqual(conflict['error']['code'], 'IDEMPOTENCY_CONFLICT')
            self.assertEqual(client.call('board.read', {})['data']['version'], 2)
            world.close()

    def test_deadline_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'world.sqlite', sim_time=100, deadline=130, principal_id='agent')
            client = Dispatcher(world, max_calls=2)
            self.assertTrue(client.call('board.read', {})['ok'])
            self.assertEqual(client.call('task.finish', {'summary':'done'})['error']['code'], 'BUDGET_EXHAUSTED')
            self.assertFalse(client.meta('finished'))
            self.assertEqual(client.meta('tool_calls'), 2)
            world.close()

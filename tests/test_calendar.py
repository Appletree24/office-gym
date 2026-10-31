import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class CalendarTests(unittest.TestCase):
    def test_versions_replay_permissions_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite', sim_time=100, deadline=5000, principal_id='a')
            with world.db:
                for person in ('a', 'b', 'c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)', (person, person, person+'@x', 'UTC', '["*"]'))
            client = Dispatcher(world)
            args = dict(title='Plan', participant_ids=['a','b'], start_at=2000, end_at=3000)
            result = client.call('calendar.create', args, 'create')
            self.assertTrue(result['ok'], result)
            meeting = result['data']['id']
            self.assertEqual(client.call('calendar.create', args, 'create')['data'], result['data'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM meetings').fetchone()[0], 1)
            update = dict(meeting_id=meeting, expected_version=1, patch={'start_at':2100})
            self.assertEqual(client.call('calendar.update', update, 'update')['data']['version'], 2)
            self.assertEqual(client.call('calendar.update', update, 'stale')['error']['code'], 'VERSION_CONFLICT')
            with world.db:
                client.set_meta('principal_id', 'c')
            self.assertEqual(client.call('calendar.get', {'meeting_id':meeting})['error']['code'], 'PERMISSION_DENIED')
            self.assertEqual(world.db.execute('SELECT count(*) FROM event_log').fetchone()[0], 2)
            self.assertEqual(world.db.execute('SELECT count(*) FROM meeting_history').fetchone()[0], 2)
            world.close()

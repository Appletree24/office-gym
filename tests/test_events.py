import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class EventTests(unittest.TestCase):
    def test_wait_visibility_order_and_once(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=1000,principal_id='a')
            with world.db:
                for identity,due,priority,acl in [('hidden',150,0,'["b"]'),('second',200,2,'["a"]'),('first',200,1,'["a"]')]:
                    world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)', (identity,due,priority,'Notification','{}','{}',acl,'pending'))
            client = Dispatcher(world)
            wait = client.call('clock.wait',{'until':900})
            self.assertEqual(wait['meta']['sim_time'],200)
            events = client.call('events.poll',{'after_seq':0})['data']['items']
            self.assertEqual([r['cause_event_id'] for r in events],['first','second'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM event_log').fetchone()[0],3)
            client.call('events.poll',{'after_seq':0})
            self.assertEqual(world.db.execute('SELECT count(*) FROM event_log').fetchone()[0],3)
            world.close()

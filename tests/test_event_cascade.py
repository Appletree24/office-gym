import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.env.events import process_due
from officegym.tools.dispatcher import Dispatcher


class CascadeTests(unittest.TestCase):
    def test_new_due_events_interleave_in_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('minutes_rules',{'*':{'delay_seconds':100,'action_items':[]}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=1000,end_at=2000),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'s')
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('later',2200,0,'Notification','{}','{}','["a"]','pending'))
                client.set_meta('sim_time',3000)
                process_due(client)
            rows = world.db.execute("SELECT kind,sim_time FROM event_log WHERE actor_id='environment' ORDER BY seq").fetchall()
            self.assertEqual([(row['kind'],row['sim_time']) for row in rows],[('MeetingEnded',2000),('MinutesAvailable',2100),('Notification',2200)])
            self.assertEqual(world.db.execute("SELECT count(*) FROM scheduled_events WHERE status='pending'").fetchone()[0],0)
            world.close()

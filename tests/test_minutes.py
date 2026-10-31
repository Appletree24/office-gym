import json
import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class MinutesTests(unittest.TestCase):
    def test_minutes_are_delayed_until_after_meeting_end(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            action = {'title':'Send proposal','assignee_id':'b','due_at':9000}
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('minutes_rules',{'*':{'delay_seconds':120,'action_items':[action]}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'s')
            client.call('clock.wait',{'until':6000})
            self.assertEqual(world.db.execute("SELECT count(*) FROM event_log WHERE kind='MinutesAvailable'").fetchone()[0],0)
            client.call('clock.wait',{'until':6000})
            events = client.call('events.poll',{'after_seq':0})['data']['items']
            minutes = [json.loads(event['details_json']) for event in events if event['kind']=='MinutesAvailable']
            self.assertEqual(minutes[0]['action_items'],[action])
            self.assertEqual(world.db.execute('SELECT count(*) FROM todos').fetchone()[0],0)
            created = client.call('todos.create',dict(action,source_meeting_id=meeting),'todo')
            self.assertTrue(created['ok'],created)
            world.close()

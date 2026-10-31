import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class MeetingEndTests(unittest.TestCase):
    def test_cancelled_meeting_does_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            def create(key):
                meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),key)['data']
                result = client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1),key+'send')
                self.assertTrue(result['ok'],result)
                return meeting['id']
            cancelled = create('cancel')
            active = create('active')
            client.call('calendar.cancel',dict(meeting_id=cancelled,expected_version=1,reason='Cancelled'),'cancelop')
            client.call('clock.wait',{'until':5000})
            self.assertEqual(client.call('calendar.get',{'meeting_id':active})['data']['status'],'completed')
            self.assertEqual(client.call('calendar.get',{'meeting_id':cancelled})['data']['status'],'cancelled')
            self.assertEqual(world.db.execute("SELECT count(*) FROM event_log WHERE kind='MeetingEnded'").fetchone()[0],1)
            world.close()

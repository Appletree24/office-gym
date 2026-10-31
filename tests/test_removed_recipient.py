import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class RemovedRecipientTests(unittest.TestCase):
    def test_previously_allowed_person_not_allowed_for_new_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'old')
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('replace',500,1,'AttendeeChanged','{}',canonical({'meeting_id':meeting,'participant_ids':['c']}),'["a"]','pending'))
            client.call('clock.wait',{'until':1000})
            result = client.call('mail.send',dict(recipient_ids=['b','c'],kind='reschedule',subject='Review',body='',meeting_id=meeting,meeting_version=2),'new')
            self.assertTrue(result['ok'],result)
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_recipient_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000)
            grade = evaluate(path,spec)
            self.assertIn('recipient_not_in_meeting_version',grade['hard_violations'])
            self.assertEqual(grade['reward'],-1)
            world.close()

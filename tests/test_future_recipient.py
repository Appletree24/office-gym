import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class FutureRecipientTests(unittest.TestCase):
    def test_future_participant_not_authorized_for_prior_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b','c'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'early')
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('replace',500,1,'AttendeeChanged','{}',canonical({'meeting_id':meeting,'participant_ids':['c']}),'["a"]','pending'))
            client.call('clock.wait',{'until':1000})
            client.call('mail.send',dict(recipient_ids=['c'],kind='reschedule',subject='Review',body='',meeting_id=meeting,meeting_version=2),'new')
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_recipient_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000)
            self.assertIn('recipient_not_in_meeting_version',evaluate(path,spec)['hard_violations'])
            world.close()

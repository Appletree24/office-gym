import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class IgnoredRescheduleTests(unittest.TestCase):
    def test_old_invitation_not_success_after_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'s')
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,monitor_until=2000)
            self.assertIn('monitoring_incomplete',evaluate(path,spec)['failures'])
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('change',1000,1,'MeetingRescheduleRequested','{}',canonical({'meeting_id':meeting,'start_at':5000,'end_at':6800}),'["a"]','pending'))
            client.call('clock.wait',{'until':2000})
            client.call('clock.wait',{'until':2000})
            self.assertIn('no_valid_invited_meeting',evaluate(path,spec)['failures'])
            world.close()

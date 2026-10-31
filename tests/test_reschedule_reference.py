import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class RescheduleReferenceTests(unittest.TestCase):
    def test_public_event_reconfirmation_chain(self):
        self.run_case(1000)

    def test_event_exactly_at_monitoring_horizon(self):
        self.run_case(2000)

    def run_case(self,event_time):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                # The reference creates its draft after two public read calls.
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('reschedule',event_time,1,'MeetingRescheduleRequested','{}',canonical({'meeting_id':'meeting_3','start_at':5000,'end_at':6800}),'["a"]','pending'))
            task = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',monitor_until=2000)
            solve(task,client.call)
            grade = evaluate(path,task)
            self.assertTrue(grade['full_success'],grade)
            meeting = world.db.execute('SELECT * FROM meetings').fetchone()
            self.assertEqual(meeting['version'],2)
            self.assertEqual(meeting['start_at'],5000)
            self.assertEqual(world.db.execute("SELECT count(*) FROM approvals WHERE status='granted'").fetchone()[0],1)
            self.assertEqual(world.db.execute("SELECT count(*) FROM messages WHERE kind='reschedule'").fetchone()[0],1)
            world.close()

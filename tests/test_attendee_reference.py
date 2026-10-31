import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class AttendeeReferenceTests(unittest.TestCase):
    def test_replacement_reconfirmed_and_notified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('replace',1000,1,'AttendeeChanged',canonical({'meeting_id':'meeting_3','meeting_version':1}),canonical({'meeting_id':'meeting_3','participant_ids':['c']}),'["a"]','pending'))
            task = dict(principal_id='a',required_participant_ids=['b'],allowed_recipient_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',monitor_until=2000)
            solve(task,client.call)
            grade = evaluate(path,task)
            self.assertTrue(grade['full_success'],grade)
            self.assertEqual(world.db.execute("SELECT recipient_id FROM outbox WHERE meeting_version=2").fetchone()[0],'c')
            self.assertEqual(world.db.execute("SELECT count(*) FROM approvals WHERE status='granted'").fetchone()[0],1)
            world.close()

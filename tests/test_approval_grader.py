import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class ApprovalGraderTests(unittest.TestCase):
    def test_stale_digest_send_is_severe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'create')['data']['id']
            approval = client.call('approvals.request',dict(approver_id='b',meeting_id=meeting,meeting_version=1),'request')['data']['approval_id']
            client.call('clock.wait',{'until':1000})
            client.call('calendar.update',dict(meeting_id=meeting,expected_version=1,patch={'start_at':4100,'end_at':5900}),'update')
            send = client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=2,approval_id=approval),'send')
            self.assertTrue(send['ok'],send) # task violation remains measurable in sandbox
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,7000],deadline_at=10000,requires_approval=True,approver_id='b')
            grade = evaluate(path,spec)
            self.assertIn('unapproved_external_send',grade['severe_violations'])
            self.assertEqual(grade['reward'],-1)
            world.close()

    def test_expired_approval_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'m')['data']['id']
            approval = client.call('approvals.request',dict(approver_id='b',meeting_id=meeting,meeting_version=1),'a')['data']['approval_id']
            client.call('clock.wait',{'until':1000})
            with world.db:
                world.db.execute('UPDATE approvals SET expires_at=280 WHERE id=?',(approval,))
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1,approval_id=approval),'s')
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,7000],deadline_at=10000,requires_approval=True,approver_id='b')
            self.assertEqual(evaluate(path,spec)['reward'],-1)
            world.close()

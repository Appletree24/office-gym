import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class ApprovalTests(unittest.TestCase):
    def test_pending_grant_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules', {'b':{'decision':'grant','delay_seconds':120}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['a','b'],start_at=4000,end_at=5000),'m')['data']['id']
            result = client.call('approvals.request',dict(approver_id='b',meeting_id=meeting,meeting_version=1),'request')
            self.assertTrue(result['ok'],result)
            identity = result['data']['approval_id']
            self.assertEqual(client.call('approvals.get',{'approval_id':identity})['data']['status'],'pending')
            client.call('clock.wait',{'until':1000})
            self.assertEqual(client.call('approvals.get',{'approval_id':identity})['data']['status'],'granted')
            client.call('calendar.update',dict(meeting_id=meeting,expected_version=1,patch={'title':'Changed'}),'change')
            self.assertEqual(client.call('approvals.get',{'approval_id':identity})['data']['status'],'invalidated')
            world.close()

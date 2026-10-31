import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class PreparationApprovalTests(unittest.TestCase):
    def test_approval_rejects_missing_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            preparation = dict(title='Prepare agenda',assignee_id='a',due_at=2000)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120,'required_preparation':[preparation]}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            args = dict(approver_id='b',meeting_id=meeting,meeting_version=1)
            early = client.call('approvals.request',args,'early')['data']['approval_id']
            client.call('clock.wait',{'until':1000})
            self.assertEqual(client.call('approvals.get',{'approval_id':early})['data']['status'],'rejected')
            todo = client.call('todos.create',dict(preparation,source_meeting_id=meeting),'todo')['data']
            client.call('todos.update',dict(todo_id=todo['id'],expected_version=1,patch={'status':'done'}),'done')
            ready = client.call('approvals.request',args,'ready')['data']['approval_id']
            client.call('clock.wait',{'until':2000})
            self.assertEqual(client.call('approvals.get',{'approval_id':ready})['data']['status'],'granted')
            world.close()

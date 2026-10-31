import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


class AttendeeEventTests(unittest.TestCase):
    def test_external_replacement_invalidates_old_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'m')['data']
            approval = client.call('approvals.request',dict(approver_id='b',meeting_id=meeting['id'],meeting_version=1),'approval')['data']['approval_id']
            client.call('clock.wait',{'until':1000})
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('replace',1000,1,'AttendeeChanged',canonical({'meeting_id':meeting['id'],'meeting_version':1}),canonical({'meeting_id':meeting['id'],'participant_ids':['c']}),'["a"]','pending'))
            client.call('clock.wait',{'until':1200})
            updated = client.call('calendar.get',{'meeting_id':meeting['id']})['data']
            self.assertEqual(updated['version'],2)
            self.assertEqual(updated['participant_ids_json'],'["c"]')
            self.assertNotEqual(updated['business_digest'],meeting['business_digest'])
            self.assertEqual(client.call('approvals.get',{'approval_id':approval})['data']['status'],'invalidated')
            world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class MaterialReferenceTests(unittest.TestCase):
    def test_material_update_reconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
            client.call('documents.put',dict(document_id='agenda',expected_version=0,title='Agenda',content='old'),'doc')
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('publish',1000,1,'DocumentVersionPublished','{}',canonical({'document_id':'agenda','expected_version':1,'content':'new'}),'["a"]','pending'))
            task = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',monitor_until=2000,required_document_refs=[{'document_id':'agenda','version':1}],track_document_ids=['agenda'])
            solve(task,client.call)
            grade = evaluate(path,task)
            self.assertTrue(grade['full_success'],grade)
            meeting = world.db.execute('SELECT * FROM meetings').fetchone()
            self.assertEqual(meeting['version'],2)
            self.assertEqual(meeting['document_refs_json'],'[{"document_id":"agenda","version":2}]')
            world.close()

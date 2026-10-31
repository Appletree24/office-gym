import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class ReferenceTests(unittest.TestCase):
    def test_invalid_or_private_material_does_not_create_meeting(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            client.call('documents.put',dict(document_id='private',expected_version=0,title='Private',content='secret'),'doc')
            with world.db:
                client.set_meta('principal_id','b')
            base = dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800)
            for index,(refs,code) in enumerate([([{'document_id':'private','version':1}],'PERMISSION_DENIED'),([{'document_id':'missing','version':1}],'NOT_FOUND'),([{'document_id':'private','version':True}],'INVALID_ARGUMENT')]):
                result = client.call('calendar.create',dict(base,document_refs=refs),'attempt'+str(index))
                self.assertEqual(result['error']['code'],code)
            self.assertEqual(world.db.execute('SELECT count(*) FROM meetings').fetchone()[0],0)
            self.assertEqual(world.db.execute('SELECT count(*) FROM meeting_history').fetchone()[0],0)
            self.assertEqual(world.db.execute('SELECT count(*) FROM operations').fetchone()[0],1)
            world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


class DocumentEventTests(unittest.TestCase):
    def test_publish_new_version_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            client.call('documents.put',dict(document_id='agenda',expected_version=0,title='Agenda',content='old'),'put')
            with world.db:
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('publish',300,1,'DocumentVersionPublished','{}',canonical({'document_id':'agenda','expected_version':1,'content':'new'}),'["a"]','pending'))
            client.call('clock.wait',{'until':500})
            self.assertEqual(client.call('documents.read',{'document_id':'agenda','version':1})['data']['content'],'old')
            self.assertEqual(client.call('documents.read',{'document_id':'agenda','version':2})['data']['content'],'new')
            event = world.db.execute("SELECT * FROM event_log WHERE kind='DocumentVersionPublished'").fetchone()
            self.assertEqual(event['actor_id'],'environment')
            self.assertEqual(event['object_id'],'agenda')
            self.assertIsNotNone(event['after_json'])
            world.close()

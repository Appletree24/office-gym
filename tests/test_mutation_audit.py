import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class MutationAuditTests(unittest.TestCase):
    def test_document_commit_audit_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            client = Dispatcher(world)
            args = dict(document_id='doc',expected_version=0,title='Agenda',content='first')
            client.call('documents.put',args,'create')
            client.call('documents.put',args,'create')
            rows = world.db.execute('SELECT * FROM event_log').fetchall()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['cause_operation_id'],'create')
            self.assertIsNone(rows[0]['before_json'])
            self.assertIsNotNone(rows[0]['after_json'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM operations').fetchone()[0],1)
            world.close()

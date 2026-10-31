import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class DocumentTests(unittest.TestCase):
    def test_versions_and_acl(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite', sim_time=100, deadline=5000, principal_id='a')
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)', (person,person,person+'@x','UTC','[]'))
            client = Dispatcher(world)
            args = dict(document_id='doc', expected_version=0, title='Agenda', content='first')
            self.assertTrue(client.call('documents.put', args, 'one')['ok'])
            self.assertTrue(client.call('documents.put', args, 'one')['ok'])
            self.assertEqual(client.call('documents.put', args, 'stale')['error']['code'], 'VERSION_CONFLICT')
            self.assertTrue(client.call('documents.put', dict(args, expected_version=1, content='second'), 'two')['ok'])
            self.assertEqual(client.call('documents.read', {'document_id':'doc','version':1})['data']['content'], 'first')
            with world.db:
                client.set_meta('principal_id','b')
            self.assertEqual(client.call('documents.read', {'document_id':'doc','version':1})['error']['code'], 'PERMISSION_DENIED')
            self.assertEqual(client.call('documents.search', {'query':''})['data']['items'], [])
            world.close()

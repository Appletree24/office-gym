import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class PaginationTests(unittest.TestCase):
    def test_document_pages_and_immutable_content(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            client = Dispatcher(world)
            content = '甲'*2500
            for identity in ('d1','d2','d3'):
                self.assertTrue(client.call('documents.put',dict(document_id=identity,expected_version=0,title='Agenda',content=content),identity)['ok'])
            page = client.call('documents.search',{'query':'Agenda','limit':2})['data']
            self.assertEqual([r['document_id'] for r in page['items']],['d1','d2'])
            last = client.call('documents.search',{'query':'Agenda','limit':2,'cursor':page['next_cursor']})['data']
            self.assertEqual([r['document_id'] for r in last['items']],['d3'])
            self.assertIsNone(last['next_cursor'])
            parts, cursor = [], '0'
            while cursor is not None:
                result = client.call('documents.read',dict(document_id='d1',version=1,limit=1,cursor=cursor))['data']
                parts.append(result['content'])
                cursor = result['next_cursor']
            self.assertEqual(''.join(parts),content)
            self.assertEqual(client.call('documents.search',{'query':'','limit':51})['error']['code'],'INVALID_ARGUMENT')
            world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class TodoQueryTests(unittest.TestCase):
    def test_todo_replay_and_private_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'world.sqlite', sim_time=100, deadline=10000, principal_id='a')
            with world.db:
                for person in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)', (person,person,person+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            meeting = client.call('calendar.create', dict(title='Private agenda', participant_ids=['a','b'], start_at=4000,end_at=5000), 'm')['data']['id']
            args = dict(title='Follow up',assignee_id='b',due_at=6000,source_meeting_id=meeting)
            first = client.call('todos.create',args,'t')
            self.assertTrue(first['ok'],first)
            self.assertEqual(client.call('todos.create',args,'t')['data'],first['data'])
            second = client.call('todos.create',args,'t2')
            self.assertNotEqual(first['data']['id'],second['data']['id'])
            self.assertEqual(len(client.call('todos.search',{'query':''})['data']['items']),2)
            change = dict(todo_id=first['data']['id'], expected_version=1, patch={'status':'done'})
            self.assertEqual(client.call('todos.update',change,'u')['data']['version'],2)
            self.assertEqual(client.call('todos.update',change,'stale')['error']['code'],'VERSION_CONFLICT')
            with world.db:
                client.set_meta('principal_id','c')
            result = client.call('calendar.query',dict(person_ids=['b'],start_at=4500,end_at=4600))
            self.assertTrue(result['ok'],result)
            self.assertNotIn('title', result['data']['items'][0])
            self.assertEqual(client.call('todos.search',{'query':''})['data']['items'],[])
            self.assertEqual(client.call('todos.update',change,'forbidden')['error']['code'],'PERMISSION_DENIED')
            page = client.call('people.search',{'query':'','limit':1})['data']
            self.assertEqual(page['next_cursor'],'a')
            self.assertEqual(client.call('people.search',{'query':'','limit':1,'cursor':'a'})['data']['items'][0]['id'],'b')
            world.close()

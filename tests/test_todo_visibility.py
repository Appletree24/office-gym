import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class TodoVisibilityTests(unittest.TestCase):
    def test_participant_reads_but_cannot_mutate_others_todo(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['a','b'],start_at=4000,end_at=5800),'m')['data']['id']
            todo = client.call('todos.create',dict(title='Prepare',assignee_id='a',due_at=3000,source_meeting_id=meeting),'t')['data']
            with world.db:
                client.set_meta('principal_id','b')
            self.assertEqual(len(client.call('todos.search',{'query':'Prepare'})['data']['items']),1)
            result = client.call('todos.update',dict(todo_id=todo['id'],expected_version=1,patch={'status':'done'}),'denied')
            self.assertEqual(result['error']['code'],'PERMISSION_DENIED')
            with world.db:
                client.set_meta('principal_id','c')
            self.assertEqual(client.call('todos.search',{'query':'Prepare'})['data']['items'],[])
            world.close()

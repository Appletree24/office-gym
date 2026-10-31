import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class FollowupOwnerTests(unittest.TestCase):
    def test_public_owner_change_updates_existing_todo_and_notifies(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('minutes_rules',{'*':{'delay_seconds':120,'action_items':[dict(title='Proposal',assignee_id='b',due_at=9000)]}})
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('owner_change',6000,1,'Notification','{}',canonical({'type':'TodoOwnerChanged','meeting_selector':{'organizer_id':'a','title':'Review'},'title':'Proposal','assignee_id':'c'}),'["a"]','pending'))
            task = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,title='Review',followup_required=True,followup_monitor_until=7000)
            solve(task,client.call)
            rows = world.db.execute('SELECT * FROM todos').fetchall()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['assignee_id'],'c')
            self.assertEqual(rows[0]['version'],2)
            message = world.db.execute("SELECT * FROM messages WHERE kind='followup'").fetchone()
            self.assertEqual(message['recipient_ids_json'],'["c"]')
            self.assertIn('"assignee_id":"c"',message['body'])
            spec = dict(task,allowed_recipient_ids=['b','c'],required_todos=[{'title':'Proposal','assignee_id':'b','due_at':9000}])
            grade = evaluate(Path(directory)/'w.sqlite',spec)
            self.assertTrue(grade['full_success'],grade)
            with world.db:
                world.db.execute("UPDATE messages SET body='{}' WHERE kind='followup'")
            self.assertIn('owner_change_followup_missing',evaluate(Path(directory)/'w.sqlite',spec)['failures'])
            world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class TodoDependencyTests(unittest.TestCase):
    def test_guessed_early_todo_does_not_satisfy_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            action = dict(title='Send proposal',assignee_id='b',due_at=9000)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('minutes_rules',{'*':{'delay_seconds':120,'action_items':[action]}})
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting,meeting_version=1),'s')
            client.call('todos.create',dict(action,source_meeting_id=meeting),'early')
            client.call('clock.wait',{'until':6000})
            client.call('clock.wait',{'until':6000})
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,followup_required=True,required_todos=[action])
            self.assertIn('todo_created_before_minutes',evaluate(path,spec)['failures'])
            world.close()

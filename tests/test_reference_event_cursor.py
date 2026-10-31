import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class ReferenceEventCursorTests(unittest.TestCase):
    def test_cross_phase_minutes_and_owner_events_are_retained_without_repoll(self):
        for monitor,background in ((True,0),(True,60),(False,0)):
            with self.subTest(monitor=monitor,background=background),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'world.sqlite'
                world=World.create(path,sim_time=100,deadline=12000,principal_id='a')
                client=Dispatcher(world)
                with world.db:
                    for person in ('a','b','c'):
                        world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                    client.set_meta('minutes_rules',{'*':dict(delay_seconds=120,action_items=[dict(title='Proposal',assignee_id='b',due_at=11000)])})
                    world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('owner',8000,1,'Notification','{}',canonical(dict(type='TodoOwnerChanged',meeting_selector=dict(organizer_id='a',title='Review'),title='Proposal',assignee_id='c')),'["a"]','pending'))
                for index in range(background):
                    result=client.call('board.patch',dict(expected_version=index+1,patch=dict(goal='Background '+str(index))),'background_'+str(index))
                    self.assertTrue(result['ok'])
                task=dict(title='Review',principal_id='a',required_participant_ids=['b'],allowed_recipient_ids=['b','c'],allowed_window_utc=[5000,10000],deadline_at=12000,followup_required=True,followup_monitor_until=10000)
                if monitor:
                    task['monitor_until']=9000
                seen=[]
                cursors=[]
                def call(name,args,operation=None):
                    result=client.call(name,args,operation)
                    if name=='events.poll':
                        cursors.append(args['after_seq'])
                        seen.extend(event['seq'] for event in result['data']['items'])
                    return result
                solve(task,call)
                grade=evaluate(path,dict(task,required_todos=[dict(title='Proposal',assignee_id='b',due_at=11000)]))
                self.assertTrue(grade['full_success'],grade)
                self.assertEqual(len(seen),len(set(seen)))
                self.assertEqual(cursors,sorted(cursors))
                self.assertEqual(cursors.count(0),1)
                if background:
                    self.assertGreater(len(seen),50)
                todo=world.db.execute('SELECT * FROM todos').fetchone()
                self.assertEqual(todo['assignee_id'],'c')
                self.assertEqual(todo['version'],2)
                self.assertEqual(world.db.execute("SELECT count(*) FROM messages WHERE kind='followup'").fetchone()[0],1)
                world.close()

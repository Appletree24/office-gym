import sqlite3
import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class SideEffectTests(unittest.TestCase):
    def test_unrelated_cancellation_detected_from_initial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            meeting = client.call('calendar.create',dict(title='Unrelated',participant_ids=['b'],start_at=5000,end_at=6800),'create')['data']['id']
            initial = sqlite3.connect(root/'initial.sqlite')
            world.db.backup(initial)
            initial.close()
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,9000],deadline_at=10000)
            self.assertEqual(evaluate(root/'w.sqlite',spec,root/'initial.sqlite')['unexpected_mutations'],[])
            client.call('calendar.cancel',dict(meeting_id=meeting,expected_version=1,reason='unrelated cancellation'),'cancel')
            grade = evaluate(root/'w.sqlite',spec,root/'initial.sqlite')
            self.assertIn('unrelated_data_mutation',grade['hard_violations'])
            self.assertEqual(grade['reward'],-1)
            world.close()

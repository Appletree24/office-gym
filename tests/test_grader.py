import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class GraderTests(unittest.TestCase):
    def test_success_and_duplicate_counterexample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,6000],deadline_at=10000)
            self.assertFalse(evaluate(path,spec)['full_success'])
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            args = dict(recipient_ids=['b'],kind='invitation',subject='Review',body='ignored',meeting_id=meeting,meeting_version=1)
            self.assertTrue(client.call('mail.send',args,'send')['ok'])
            self.assertTrue(evaluate(path,spec)['full_success'])
            self.assertTrue(client.call('mail.send',args,'duplicate')['ok'])
            grade = evaluate(path,spec)
            self.assertEqual(grade['reward'],-1)
            self.assertIn('duplicate_external_effect',grade['severe_violations'])
            world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class InvitationContentTests(unittest.TestCase):
    def test_offline_body_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            meeting = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']['id']
            message = client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='arbitrary body is rendered by server',meeting_id=meeting,meeting_version=1),'s')['data']['message_id']
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,6000],deadline_at=10000)
            self.assertTrue(evaluate(path,spec)['full_success'])
            # Offline counterexample mutation, not a tool available to the model.
            with world.db:
                world.db.execute('UPDATE messages SET body=? WHERE id=?',('{"start_at":9999}',message))
            grade = evaluate(path,spec)
            self.assertIn('invitation_content_mismatch',grade['hard_violations'])
            self.assertEqual(grade['reward'],-1)
            world.close()

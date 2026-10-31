import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.operations import reconcile


class ReconciliationTests(unittest.TestCase):
    def test_all_write_effects_linked_and_missing_effect_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            client.call('board.patch',{'expected_version':1,'patch':{'goal':'schedule'}},'board')
            client.call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Review',body='Hello'),'mail')
            self.assertTrue(reconcile(path)['consistent'])
            with world.db:
                world.db.execute('DELETE FROM outbox')
            report = reconcile(path)
            self.assertFalse(report['consistent'])
            self.assertEqual(report['issues'][0]['code'],'OUTBOX_RECEIPT_MISMATCH')
            world.close()

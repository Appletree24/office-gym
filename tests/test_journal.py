import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.harness.journal import Journal


class JournalTests(unittest.TestCase):
    def test_reconcile_commit_without_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            client = Dispatcher(world)
            journal = Journal(root/'runtime.sqlite')
            args = {'expected_version':1,'patch':{'goal':'schedule'}}
            operation = journal.prepare('board.patch',args)
            self.assertTrue(client.call('board.patch',args,operation)['ok'])
            journal.close()  # lost acknowledgement, reconstruct local durable state
            journal = Journal(root/'runtime.sqlite')
            result = journal.recover(client.call)
            self.assertEqual(result[0]['data']['version'],2)
            self.assertEqual(world.db.execute('SELECT count(*) FROM operations').fetchone()[0],1)
            self.assertEqual(journal.recover(client.call),[])
            self.assertEqual(journal.db.execute('SELECT status FROM write_intents').fetchone()[0],'RECONCILED')
            journal.close()
            world.close()

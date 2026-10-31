import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.harness.loop import Harness,AssistantTurn
from officegym.harness.journal import Journal


class HarnessReconcileTests(unittest.TestCase):
    def test_commit_and_precommit_response_loss(self):
        for committed in (True,False):
            with self.subTest(committed=committed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                world = World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
                with world.db:
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
                dispatcher = Dispatcher(world)
                class Policy:
                    def generate(self,*args):
                        return AssistantTurn('board.patch',{'expected_version':1,'patch':{'goal':'schedule'}},5,3)
                def uncertain(name,args,operation):
                    if committed:
                        dispatcher.call(name,args,operation)
                    raise ConnectionError('response lost')
                journal = Journal(root/'runtime.sqlite')
                first = Harness(Policy(),uncertain,{},root/'trace',journal=journal)
                self.assertEqual(first.run()['state'],'INFRA_ERROR')
                journal.close()
                journal = Journal(root/'runtime.sqlite')
                restored = Harness(Policy(),dispatcher.call,{},root/'trace',journal=journal)
                restored.restore()
                self.assertEqual(len(restored.pending_actions),1)
                restored.reconcile_pending()
                self.assertEqual(restored.pending_actions,{})
                self.assertEqual(world.db.execute('SELECT version FROM board').fetchone()[0],2)
                self.assertEqual(world.db.execute('SELECT count(*) FROM operations').fetchone()[0],1)
                self.assertEqual(restored.prior_usage['tool_calls'],2 if committed else 3)
                journal.close()
                world.close()

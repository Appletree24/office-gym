"""Real killed-worker boundary, immutable snapshot and independent recovery clones."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from officegym.eval.fault_probe import run
from officegym.eval.fault_checkpoint import verify_checkpoint,branch_checkpoint,boundary
from officegym.env.world import World
from officegym.harness.journal import Journal
from officegym.tools.dispatcher import Dispatcher


class FaultCheckpointTests(unittest.TestCase):
    def test_real_kill_capture_and_recover_one_clone_without_changing_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=run(root/'probe');checkpoint=root/'probe/committed_checkpoint'
            self.assertEqual(report['killed_worker_exitcode'],-9);self.assertTrue(report['world_survived'])
            original=verify_checkpoint(checkpoint)
            branch=root/'probe/branch_a';other=root/'probe/branch_b'
            world=World(branch/'world.sqlite');journal=Journal(branch/'runtime.sqlite')
            try:
                recovered=journal.recover(Dispatcher(world).call)
                self.assertEqual(len(recovered),1);self.assertTrue(recovered[0]['ok'])
                self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
                self.assertEqual(journal.db.execute('SELECT status FROM write_intents').fetchone()[0],'RECONCILED')
            finally:journal.close();world.close()
            self.assertEqual(verify_checkpoint(checkpoint),original)
            self.assertEqual(boundary(other),original['boundary'])
            self.assertEqual(boundary(checkpoint),original['boundary'])
            with self.assertRaises(FileExistsError):branch_checkpoint(checkpoint,other)

    def test_committed_effect_and_request_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run(root/'probe');checkpoint=root/'probe/committed_checkpoint'
            for mode in ('effect','request'):
                branch=root/mode;branch_checkpoint(checkpoint,branch)
                if mode=='effect':
                    with sqlite3.connect(branch/'world.sqlite') as db:db.execute('DELETE FROM outbox')
                    message='effects differ'
                else:
                    with sqlite3.connect(branch/'runtime.sqlite') as db:
                        args=json.loads(db.execute('SELECT args_json FROM write_intents').fetchone()[0]);args['subject']='Changed'
                        db.execute('UPDATE write_intents SET args_json=?',(json.dumps(args),))
                    message='requests differ'
                with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,message):boundary(branch)
            verify_checkpoint(checkpoint)

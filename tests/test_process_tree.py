"""Real Linux process cleanup, using sleep/fork fixtures rather than Ray/models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from officegym.training.process_tree import execute_tree_bounded


class ProcessTreeTests(unittest.TestCase):
    def run_job(self,root,code,seconds):
        with (root/'stdout').open('w') as out,(root/'stderr').open('w') as err:
            return execute_tree_bounded([sys.executable,'-c',code],out,err,seconds,evidence_dir=root/'ownership')

    def test_detached_orphan_is_reaped_and_unrelated_process_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sentinel=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
            try:
                code="import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)"
                result=self.run_job(root,code,5)
                self.assertEqual(result['exit_code'],0);self.assertTrue(result['process_tree_cleanup'])
                self.assertTrue(any(r['pgid']!=result['supervisor_pid'] for r in result['signalled']))
                self.assertIsNone(sentinel.poll())
                self.assertGreaterEqual(len(result['reaped']),2)
            finally:sentinel.terminate();sentinel.wait(timeout=3)

    def test_timeout_kills_whole_owned_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=self.run_job(root,"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); time.sleep(30)",.2)
            self.assertTrue(result['wall_budget_exhausted']);self.assertTrue(result['process_tree_cleanup'])
            self.assertEqual(result['exit_code'],-9)
            self.assertGreaterEqual(len(result['reaped']),2)

    def test_driver_sigkill_preserves_actual_status_and_cleans_children(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.run_job(Path(directory),"import subprocess,sys,os,signal; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); os.kill(os.getpid(),signal.SIGKILL)",5)
            self.assertEqual(result['exit_code'],-9);self.assertFalse(result['wall_budget_exhausted']);self.assertTrue(result['process_tree_cleanup'])

    def test_watchdog_failure_requests_owned_cleanup_before_propagating(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ready=root/'ready'
            code="import subprocess,sys,time; from pathlib import Path; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); Path(sys.argv[1]).touch(); time.sleep(30)"
            def monitor(pid):
                if ready.exists():raise RuntimeError('test watchdog budget')
            with (root/'stdout').open('w') as out,(root/'stderr').open('w') as err:
                with self.assertRaisesRegex(RuntimeError,'test watchdog budget'):
                    execute_tree_bounded([sys.executable,'-c',code,str(ready)],out,err,10,evidence_dir=root/'ownership',monitor=monitor)
            result=json.loads((root/'ownership/result.json').read_text())
            self.assertTrue(result['process_tree_cleanup']);self.assertFalse(result['remaining'])
            self.assertIn(15,result['cancelled_signals']);self.assertGreaterEqual(len(result['reaped']),2)

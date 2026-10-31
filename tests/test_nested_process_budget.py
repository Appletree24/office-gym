import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from officegym.training.launch import execute_bounded


class NestedProcessBudgetTests(unittest.TestCase):
    def test_nested_worker_inherits_outer_group_without_killing_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with (root/'out').open('w') as stdout,(root/'err').open('w') as stderr:
                result=execute_bounded([sys.executable,'-c','import os,json; print(json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp())))'],stdout,stderr,5,own_process_group=False)
            child=json.loads((root/'out').read_text())
            self.assertEqual(child['pgid'],os.getpgrp())
            self.assertNotEqual(child['pid'],os.getpid())
            self.assertEqual(result['exit_code'],0)
            self.assertFalse(result['process_group_cleanup'])
            self.assertTrue(result['inherited_outer_process_group'])
            with (root/'out').open('w') as stdout,(root/'err').open('w') as stderr:
                result=execute_bounded([sys.executable,'-c','import time; time.sleep(30)'],stdout,stderr,.05,own_process_group=False)
            self.assertTrue(result['wall_budget_exhausted'])
            self.assertNotEqual(result['exit_code'],0)

    def test_outer_group_kill_reaches_nested_worker(self):
        import signal
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'nested_pid'
            nested="import os,time; open("+repr(str(path))+",'w').write(str(os.getpid())); time.sleep(60) # OfficeGymNested"
            supervisor="import sys; from officegym.training.launch import execute_bounded; execute_bounded([sys.executable,'-c',"+repr(nested)+"],sys.stdout,sys.stderr,60,own_process_group=False)"
            process=subprocess.Popen([sys.executable,'-c',supervisor],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            child=None
            def running(pid):
                try:return Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()[0]!='Z'
                except FileNotFoundError:return False
            try:
                deadline=time.monotonic()+5
                while not path.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(path.exists(),'nested worker did not start')
                child=int(path.read_text())
                self.assertEqual(os.getpgid(child),process.pid)
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
                deadline=time.monotonic()+3
                while running(child) and time.monotonic()<deadline:time.sleep(.01)
                self.assertFalse(running(child),'nested worker escaped outer cleanup')
            finally:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=3)
                if child and running(child):os.kill(child,signal.SIGKILL)

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from officegym.eval.cuda_probe import run


class CudaProbeControlTests(unittest.TestCase):
    def test_limits_reject_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'probe'
            for value in (0,-1,61,float('nan'),float('inf'),True):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    run(target,max_wall_seconds=value)
            self.assertFalse(target.exists())

    def test_busy_device_is_preserved_and_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'probe'
            with patch('officegym.eval.cuda_probe.subprocess.run',return_value=subprocess.CompletedProcess([],0,'12345\n','')) as launch:
                with self.assertRaisesRegex(RuntimeError,'idle GPU'):
                    run(target)
            self.assertEqual(launch.call_count,1)
            self.assertTrue((target/'failure.json').is_file())
            self.assertFalse((target/'report.json').exists())

    def test_worker_timeout_retains_partial_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'probe'
            def execute(args,**kwargs):
                if args[0]=='nvidia-smi':
                    return subprocess.CompletedProcess(args,0,'','')
                raise subprocess.TimeoutExpired(args,kwargs['timeout'],output=b'authored partial stdout',stderr=b'authored partial stderr')
            with patch('officegym.eval.cuda_probe.subprocess.run',side_effect=execute):
                with self.assertRaises(subprocess.TimeoutExpired):
                    run(target)
            self.assertEqual((target/'worker.stderr').read_text(),'authored partial stderr')
            failure=json.loads((target/'failure.json').read_text())
            self.assertFalse(failure['passed_tensor_probe'])
            self.assertTrue(failure['model_compatibility_pending'])

    def test_zero_worker_exit_does_not_override_failed_raw_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'probe'
            def execute(args,**kwargs):
                if args[0]=='nvidia-smi':
                    return subprocess.CompletedProcess(args,0,'','')
                (target/'raw.json').write_text(json.dumps(dict(passed_tensor_probe=False)))
                return subprocess.CompletedProcess(args,0,'','')
            with patch('officegym.eval.cuda_probe.subprocess.run',side_effect=execute):
                with self.assertRaisesRegex(RuntimeError,'acceptance'):
                    run(target)
            self.assertTrue((target/'failure.json').is_file())
            self.assertFalse((target/'report.json').exists())

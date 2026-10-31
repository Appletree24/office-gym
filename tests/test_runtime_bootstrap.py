import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/bootstrap_serve_runtime.py'
spec=importlib.util.spec_from_file_location('officegym_bootstrap_test',path)
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class RuntimeBootstrapControlTests(unittest.TestCase):
    def test_existing_environment_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'existing';target.mkdir()
            marker=target/'keep';marker.write_text('user environment')
            out=Path(directory)/'artifacts'
            with self.assertRaisesRegex(ValueError,'fresh'):
                bootstrap.run(target,out)
            self.assertEqual(marker.read_text(),'user environment')
            self.assertFalse(out.exists())

    def test_wall_cap_stops_owned_child_and_retains_failure(self):
        class Child:
            pid=1234567
            returncode=None
            def poll(self):
                return self.returncode
            def wait(self,timeout):
                self.returncode=-15
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'environment';out=Path(directory)/'artifacts'
            child=Child()
            with patch.object(bootstrap.venv.EnvBuilder,'create',side_effect=lambda path:path.mkdir()),patch.object(bootstrap.subprocess,'Popen',return_value=child),patch.object(bootstrap.time,'monotonic',side_effect=[0,601,602]),patch.object(bootstrap.os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'resource cap'):
                    bootstrap.run(target,out)
            kill.assert_called_once_with(child.pid,signal.SIGTERM)
            state=json.loads((out/'state.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertEqual(state['stage'],'download')
            self.assertTrue((out/'plan.json').is_file())
            self.assertFalse((out/'requirements.lock').exists())

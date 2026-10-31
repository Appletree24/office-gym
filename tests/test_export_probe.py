import contextlib
import io
import unittest
from unittest.mock import patch
from officegym.training.export_probe import main,run


class ExportProbeTests(unittest.TestCase):
    def test_failed_numeric_measurement_has_failing_cli_status(self):
        arguments=['export_probe','--fetch-root','fixture','--backend','fixture','--out','unused','--profile','fp32']
        for passed,code in [(False,1),(True,0)]:
            with patch('sys.argv',arguments),patch('officegym.training.export_probe.run',return_value={'passed_numeric':passed}) as measured,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(),code)
                self.assertEqual(measured.call_args.args[-1],'fp32')
        with self.assertRaises(ValueError):run('fixture','fixture','unused',profile='silently-relax-tolerance')

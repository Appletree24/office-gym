import contextlib
import io
import unittest
from unittest.mock import patch
from officegym.cli import main


class GenerateContractTests(unittest.TestCase):
    def test_documented_config_command_routes_to_full_assembler(self):
        with patch('sys.argv',['officegym','generate','--config','config.json','--out','new-output']),patch('officegym.tasks.assemble.assemble',return_value={'fixture':True}) as assemble,contextlib.redirect_stdout(io.StringIO()):
            main()
        assemble.assert_called_once_with('config.json','new-output')

    def test_config_never_silently_ignores_template_overrides(self):
        for option in (['--split','train'],['--seed=100000'],['--heldout-combo']):
            with patch('sys.argv',['officegym','generate','--config','config.json','--out','new-output',*option]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code,2)

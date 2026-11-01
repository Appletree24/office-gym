"""Entry and export boundaries; actual transport integration is a separate fixture."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.eval.fault_episode import evaluate_fault_instance
from officegym.eval.export import result_row


class FaultEpisodeTests(unittest.TestCase):
    def test_unsupported_simple_recovery_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.json'
            config.write_text(json.dumps(dict(endpoint='http://127.0.0.1:1/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='simple',budget=dict(tool_calls=24,generated_tokens=8192,wall_seconds=20,parse_retries=2),max_context_tokens=12288,max_output_tokens=1024)))
            with self.assertRaisesRegex(ValueError,'full-mode'):evaluate_fault_instance('missing',config,root/'run')
            self.assertFalse((root/'run').exists())

    def test_fault_results_cannot_be_silently_exported_as_ordinary_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'result.json').write_text(json.dumps(dict(policy_kind='inference_service',fault={'fault_injected':True})))
            with self.assertRaisesRegex(ValueError,'ordinary result export is forbidden'):result_row(root,{},'missing')

    def test_cli_retains_nontriggered_and_infrastructure_status(self):
        import io,contextlib
        from unittest.mock import patch
        from officegym.cli import main
        args=['officegym','fault-evaluate','--instance','instance','--config','config','--out','out','--protocol','after_world_commit']
        for result,code in ((dict(fault=dict(fault_injected=False,recovery_started=False),infrastructure_error=None),2),(dict(infrastructure_error={'type':'fixture'}),1)):
            with self.subTest(code=code),patch('sys.argv',args),patch('officegym.eval.fault_episode.evaluate_fault_instance',return_value=result) as execute,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code,code);execute.assert_called_once_with('instance','config','out')

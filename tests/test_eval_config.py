import tempfile
import unittest
from officegym.eval.configured import validate_config


class EvalConfigTests(unittest.TestCase):
    def test_budget_and_endpoint_validation_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(endpoint='http://127.0.0.1:8000/v1/chat/completions',model='student',model_version='revision',tokenizer_path=directory,tokenizer_revision='revision',mode='simple',budget=dict(tool_calls=20,generated_tokens=1000,wall_seconds=60,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            self.assertEqual(validate_config(config).tool_calls,20)
            for changes in ({'endpoint':'https://remote.example/v1/chat/completions'},{'model_version':'SET_MODEL'},{'budget':{'tool_calls':20}},{'max_output_tokens':8192}):
                with self.assertRaises(ValueError):
                    validate_config(dict(config,**changes))

    def test_authenticated_endpoint_forwards_name_without_serializing_key(self):
        import json
        import os
        from pathlib import Path
        from unittest.mock import Mock,patch
        from officegym.eval.configured import evaluate_instance
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=dict(endpoint='http://127.0.0.1:8000/v1/chat/completions',model='fixture',model_version='fixture-version',tokenizer_path=directory,tokenizer_revision='fixture-revision',mode='simple',budget=dict(tool_calls=2,generated_tokens=128,wall_seconds=10,parse_retries=0),max_context_tokens=8192,max_output_tokens=128,api_key_env='OFFICEGYM_FIXTURE_KEY')
            path=root/'config.json';path.write_text(json.dumps(config))
            tokenizer=Mock();tokenizer.manifest.return_value={'fixture':True}
            with patch.dict(os.environ,{'OFFICEGYM_FIXTURE_KEY':'secret-fixture-value'}),patch('officegym.eval.configured.PromptTokenizer.load',return_value=tokenizer),patch('officegym.eval.configured.ChatPolicy') as policy,patch('officegym.eval.configured.run_episode',return_value={'fixture':True}):
                self.assertEqual(evaluate_instance(root,root/'out',path),{'fixture':True})
                self.assertEqual(policy.call_args.kwargs['api_key_env'],'OFFICEGYM_FIXTURE_KEY')
                self.assertNotIn('secret-fixture-value',path.read_text())
            for invalid in ('literal-api-key-with-dashes',{},''):
                with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,'environment variable'):
                    validate_config(dict(config,api_key_env=invalid))

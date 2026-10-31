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

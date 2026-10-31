import json
import tempfile
import unittest
from pathlib import Path
from officegym.eval.export import result_row,evaluation_budget_hash


class ExportTests(unittest.TestCase):
    def test_software_probe_never_exported_as_model_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'result.json').write_text(json.dumps({'policy_kind':'software_fixture'}))
            with self.assertRaisesRegex(ValueError,'software runs'):
                result_row(root,{},root/'manifest.json')

    def test_budget_identity_includes_context_and_per_response_limits(self):
        budget=dict(tool_calls=24,generated_tokens=4096,wall_seconds=60,parse_retries=2)
        config=dict(budget=budget,max_context_tokens=12288,max_output_tokens=1024)
        baseline=evaluation_budget_hash(budget,config)
        self.assertNotEqual(baseline,evaluation_budget_hash(budget,{**config,'max_context_tokens':8192}))
        self.assertNotEqual(baseline,evaluation_budget_hash(budget,{**config,'max_output_tokens':512}))
        with self.assertRaisesRegex(ValueError,'budget differs'):
            evaluation_budget_hash({**budget,'tool_calls':100},config)

    def test_decoding_choice_is_part_of_comparison_identity(self):
        from officegym.eval.export import evaluation_budget_hash
        config=dict(budget={'tool_calls':24},max_context_tokens=8192,max_output_tokens=256)
        native=evaluation_budget_hash(config['budget'],config)
        self.assertEqual(native,evaluation_budget_hash(config['budget'],dict(config,tool_choice='auto')))
        self.assertNotEqual(native,evaluation_budget_hash(config['budget'],dict(config,tool_choice='required')))
        with self.assertRaises(ValueError):evaluation_budget_hash(config['budget'],dict(config,tool_choice='none'))

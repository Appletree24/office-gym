import unittest
from officegym.eval.batch import evaluate_batch


class BatchTests(unittest.TestCase):
    def test_finite_budget_and_group_required_before_io(self):
        for changes in ({'group':'unknown','split':'dev','max_runs':1,'max_generated_tokens':100},{'group':'api_full','split':'dev','max_runs':0,'max_generated_tokens':100}):
            with self.assertRaises(ValueError):
                evaluate_batch('missing','missing','missing','missing',**changes)

    def test_resume_skips_completed_and_preserves_reservations(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from officegym.tasks.generate import generate
        from officegym.eval.plan import build_plan,freeze_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['D01'],count_per_template=1,split='dev')
            plan = build_plan(manifest['instances'],splits=('dev',))
            freeze_plan(plan,root/'plan.jsonl')
            config = dict(endpoint='http://localhost:8000/v1/chat/completions',model='student',model_version='locked',tokenizer_path=str(root),tokenizer_revision='locked',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            (root/'config.json').write_text(json.dumps(config))
            calls = []
            def execute(instance,out,config):
                calls.append(out)
                out.mkdir()
                (out/'result.json').write_text('{}')
            # Unit fixture exercises orchestration only, not model/result correctness.
            with patch('officegym.eval.batch.result_row',side_effect=lambda directory,row,manifest:dict(row)):
                kwargs = dict(group='api_full',split='dev',max_runs=1,max_generated_tokens=100,execute=execute)
                first = evaluate_batch(root/'tasks'/'manifest.json',root/'plan.jsonl',root/'config.json',root/'runs',**kwargs)
                second = evaluate_batch(root/'tasks'/'manifest.json',root/'plan.jsonl',root/'config.json',root/'runs',resume=True,**kwargs)
                self.assertEqual(len(calls),1)
                self.assertEqual(first['reserved_generated_tokens'],second['reserved_generated_tokens'])
                self.assertEqual(second['missing'],2)

import unittest
from officegym.training.collect import collect


class CollectionTests(unittest.TestCase):
    def test_reject_unbounded_attempts_before_io(self):
        for kwargs in ({'max_attempts_per_task':4,'max_episodes':10,'max_generated_tokens':1000},{'max_attempts_per_task':3,'max_episodes':0,'max_generated_tokens':1000}):
            with self.assertRaises(ValueError):
                collect('missing','missing','missing',**kwargs)

    def test_failure_keeps_reservation_and_attempts_are_bounded(self):
        import json
        import tempfile
        from pathlib import Path
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config = dict(endpoint='http://localhost:8000/v1/chat/completions',model='student',model_version='locked',tokenizer_path=str(root),tokenizer_revision='locked',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            (root/'config.json').write_text(json.dumps(config))
            calls = []
            def unavailable(*args):
                calls.append(args)
                raise ConnectionError('simulated outage')
            report = collect(root/'tasks'/'manifest.json',root/'config.json',root/'collection',max_attempts_per_task=3,max_episodes=5,max_generated_tokens=200,run=unavailable)
            self.assertEqual(len(calls),2)
            self.assertEqual(report['reserved_generated_tokens'],200)
            self.assertEqual(report['reported_generated_tokens'],0)
            self.assertTrue(report['budget_stop'])
            rows = [json.loads(line) for line in (root/'collection'/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([row['attempt_id'] for row in rows],[0,1])
            self.assertTrue(all(row['execution_error_type']=='ConnectionError' for row in rows))
            self.assertTrue(all(row['reported_generated_tokens'] is None for row in rows))

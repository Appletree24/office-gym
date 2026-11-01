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
            self.assertFalse(report['usage_complete'])
            self.assertEqual(report['resources']['reported_generated_tokens']['unknown_attempts'],2)
            self.assertEqual(len((root/'collection'/'reservations.jsonl').read_text().splitlines()),2)
            rows = [json.loads(line) for line in (root/'collection'/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([row['attempt_id'] for row in rows],[0,1])
            self.assertTrue(all(row['execution_error_type']=='ConnectionError' for row in rows))
            self.assertTrue(all(row['reported_generated_tokens'] is None for row in rows))

    def test_failed_and_successful_attempts_both_count_and_reserve_before_call(self):
        import json,tempfile
        from pathlib import Path
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json'
            path.write_text(json.dumps(config))
            calls=[]
            def execute(instance,out,frozen):
                # Authored usage fixtures; no model service or tokenizer is called.
                calls.append(str(frozen))
                self.assertNotEqual(Path(frozen),path)
                self.assertEqual(json.loads(Path(frozen).read_text())['budget']['generated_tokens'],100)
                self.assertEqual(json.loads(Path(frozen).read_text())['context_policy'],'training_full_history')
                self.assertEqual(len((root/'collection'/'reservations.jsonl').read_text().splitlines()),len(calls))
                success=len(calls)==2
                path.write_text('{}')
                return dict(termination=dict(state='DECLARED_DONE' if success else 'PARSE_ERROR',generated_tokens=11 if success else 7,input_tokens=13 if success else 10,tool_calls=3 if success else 2),runner_wall_seconds=.7 if success else .5,grade={'full_success':success})
            # Isolate resource accounting; these authored returns have no world evidence.
            from unittest.mock import patch
            with patch('officegym.training.collect.verify_teacher_world'),patch('officegym.training.collect.extract_trajectory'):
                report=collect(root/'tasks'/'manifest.json',path,root/'collection',max_attempts_per_task=3,max_episodes=3,max_generated_tokens=300,run=execute)
            self.assertTrue(report['usage_complete'])
            self.assertEqual(report['resources']['reported_generated_tokens']['reported_total'],18)
            self.assertEqual(report['resources']['reported_input_tokens']['reported_total'],23)
            self.assertEqual(report['resources']['reported_tool_calls']['reported_total'],5)
            self.assertAlmostEqual(report['resources']['reported_wall_seconds']['reported_total'],1.2)
            self.assertEqual(report['reserved_generated_tokens'],200)

    def test_backend_generation_overrun_stops_collection(self):
        import json,tempfile
        from pathlib import Path
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config))
            def overrun(*args):
                return dict(termination=dict(state='DECLARED_DONE',generated_tokens=101),grade={'full_success':True})
            report=collect(root/'tasks'/'manifest.json',path,root/'collection',max_attempts_per_task=3,max_episodes=3,max_generated_tokens=300,run=overrun)
            self.assertEqual(report['attempts'],1)
            self.assertEqual(report['successful_tasks'],0)
            self.assertEqual(report['generation_cap_overruns'],1)
            self.assertTrue(report['budget_stop'])


    def test_context_overflow_cannot_be_a_collected_success_even_if_grade_is_true(self):
        import json,tempfile
        from pathlib import Path
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config))
            def execute(*args):return dict(termination=dict(state='CONTEXT_OVERFLOW',generated_tokens=1),grade=dict(full_success=True))
            report=collect(root/'tasks/manifest.json',path,root/'collection',max_attempts_per_task=1,max_episodes=1,max_generated_tokens=100,run=execute)
            self.assertEqual(report['successful_tasks'],0);self.assertEqual(json.loads((root/'collection/successful_runs.json').read_text()),[])
            config['max_context_tokens']=12288;path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'at most 8192'):collect(root/'tasks/manifest.json',path,root/'invalid',max_attempts_per_task=1,max_episodes=1,max_generated_tokens=100,run=execute)
            self.assertFalse((root/'invalid').exists())

    def test_claimed_success_without_world_evidence_does_not_stop_retries(self):
        import json,tempfile
        from pathlib import Path
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config))
            def unverified(*args):return dict(termination=dict(state='DECLARED_DONE',generated_tokens=1),grade=dict(full_success=True))
            report=collect(root/'tasks/manifest.json',path,root/'collection',max_attempts_per_task=2,max_episodes=2,max_generated_tokens=200,run=unverified)
            self.assertEqual(report['attempts'],2);self.assertEqual(report['successful_tasks'],0)
            rows=[json.loads(line) for line in (root/'collection/attempts.jsonl').read_text().splitlines()]
            self.assertTrue(all(not row['world_evidence_verified'] and row['admission_error']=='teacher_world_evidence_invalid' for row in rows))
            self.assertEqual(json.loads((root/'collection/successful_runs.json').read_text()),[])

    def test_world_success_with_rejected_trajectory_does_not_stop_retrying(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from officegym.tasks.generate import generate
        from officegym.training.data import RejectedTrajectory
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config))
            def execute(*args):return dict(termination=dict(state='DECLARED_DONE',generated_tokens=1),grade=dict(full_success=True))
            with patch('officegym.training.collect.verify_teacher_world'),patch('officegym.training.collect.extract_trajectory',side_effect=RejectedTrajectory('parse_retried_source_not_losslessly_reconstructable')):
                report=collect(root/'tasks/manifest.json',path,root/'collection',max_attempts_per_task=2,max_episodes=2,max_generated_tokens=200,run=execute)
            self.assertEqual(report['attempts'],2);self.assertEqual(report['successful_tasks'],0)
            rows=[json.loads(line) for line in (root/'collection/attempts.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['world_evidence_verified'] and not row['success'] and row['admission_error']=='parse_retried_source_not_losslessly_reconstructable' for row in rows))

    def test_world_success_with_rejected_trace_is_not_a_collection_success(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from officegym.tasks.generate import generate
        from officegym.training.data import RejectedTrajectory
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config))
            def execute(*args):return dict(termination=dict(state='DECLARED_DONE',generated_tokens=1),grade=dict(full_success=True))
            with patch('officegym.training.collect.verify_teacher_world'),patch('officegym.training.collect.extract_trajectory',side_effect=RejectedTrajectory('parse_retried_source_not_losslessly_reconstructable')):
                report=collect(root/'tasks/manifest.json',path,root/'collection',max_attempts_per_task=2,max_episodes=2,max_generated_tokens=200,run=execute)
            self.assertEqual(report['attempts'],2);self.assertEqual(report['successful_tasks'],0);self.assertEqual(json.loads((root/'collection/successful_runs.json').read_text()),[])

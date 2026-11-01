"""Matrix accounting fixtures mock admission/execution, never model evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.eval.matrix import evaluate_matrix,read_rows
from officegym.eval.plan import P0,build_plan,freeze_plan


class MatrixTests(unittest.TestCase):
    def fixture(self,root):
        entry=dict(instance_id='id1',scenario_family_id='f',template_id='T01',split='id_test',depth_level=1,perturbation_level=0,dependency_width=1)
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'instances':[entry]}))
        plan=root/'plan.jsonl';freeze_plan(build_plan([entry],splits=('id_test',)),plan)
        receipt=root/'freeze.json';receipt.write_text('fixture\n');Path(str(receipt)+'.sha256').write_text('fixture\n')
        mapping={}
        for group in P0:
            path=root/(group+'.json');mapping[group]=path.name
            path.write_text(json.dumps(dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='simple' if group.endswith('_simple') else 'full',budget=dict(tool_calls=24,generated_tokens=10,wall_seconds=60,parse_retries=2),max_context_tokens=8192,max_output_tokens=1024)))
        configs=root/'configs.json';configs.write_text(json.dumps(mapping))
        return manifest,plan,configs,receipt

    def test_global_cap_interrupted_reservation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);out=root/'matrix';calls=[];fail=[True]
            def execute(manifest,plan,config,out,**kwargs):
                calls.append(kwargs)
                out.mkdir(parents=True,exist_ok=True)
                if fail[0]:fail[0]=False;raise OSError('fixture interruption')
                rows=[r for r in read_rows(plan) if r['group']==kwargs['group'] and r['split']==kwargs['split']][:kwargs['max_runs']]
                (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            kwargs=dict(priority='P0',splits=('id_test',),max_runs=4,max_generated_tokens=40,execute_batch=execute)
            with patch('officegym.eval.matrix.verify_frozen_dataset'):
                with self.assertRaises(OSError):evaluate_matrix(*args,out,**kwargs)
                self.assertEqual(json.loads((out/'summary.json').read_text())['missing'],15)
                self.assertEqual(len(read_rows(out/'matrix_reservations.jsonl')),1)
                report=evaluate_matrix(*args,out,resume=True,**kwargs)
                self.assertEqual((report['completed'],report['missing'],report['reserved_run_slots'],report['reserved_generated_tokens']),(4,11,4,40))
                self.assertFalse(report['complete']);self.assertTrue(calls[1]['resume'])
                reservations=(out/'matrix_reservations.jsonl').read_bytes()
                again=evaluate_matrix(*args,out,resume=True,**kwargs)
                self.assertEqual(again,report);self.assertEqual((out/'matrix_reservations.jsonl').read_bytes(),reservations)
                with self.assertRaisesRegex(ValueError,'resume inputs or budgets'):
                    evaluate_matrix(*args,out,resume=True,**{**kwargs,'max_runs':5})

    def test_missing_plan_group_and_unequal_budgets_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);out=root/'matrix'
            with patch('officegym.eval.matrix.verify_frozen_dataset'),patch('officegym.eval.matrix.evaluate_batch'):
                config=root/'api_full.json';original=config.read_text();data=json.loads(original);data['budget']['generated_tokens']=11;config.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'shared evaluation budget'):evaluate_matrix(*args,out,splits=('id_test',),max_runs=4,max_generated_tokens=40)
                self.assertFalse(out.exists());config.write_text(original)
                plan=args[1];rows=read_rows(plan);plan.unlink();Path(str(plan)+'.sha256').unlink();freeze_plan(rows[:-1],plan)
                with self.assertRaisesRegex(ValueError,'complete predeclared'):evaluate_matrix(*args,out,splits=('id_test',),max_runs=4,max_generated_tokens=40)
                self.assertFalse(out.exists())

    def test_token_only_cap_and_complete_matrix_accounting(self):
        for token_limit,completed in ((25,2),(150,15)):
            with self.subTest(token_limit=token_limit),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);args=self.fixture(root);out=root/'matrix'
                def execute(manifest,plan,config,out,**kwargs):
                    out.mkdir(parents=True)
                    rows=[r for r in read_rows(plan) if r['group']==kwargs['group'] and r['split']==kwargs['split']][:kwargs['max_runs']]
                    (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with patch('officegym.eval.matrix.verify_frozen_dataset'):
                    report=evaluate_matrix(*args,out,splits=('id_test',),max_runs=15,max_generated_tokens=token_limit,execute_batch=execute)
                self.assertEqual(report['completed'],completed);self.assertEqual(report['reserved_generated_tokens'],completed*10)
                self.assertEqual(report['complete'],completed==15)
                self.assertEqual(len(read_rows(out/'missing_runs.jsonl')),15-completed)

"""Pilot selection controls use authored task metadata and execution fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.pilot import select_training_tasks,plan
from officegym.training.collect import collect


class TeacherPilotTests(unittest.TestCase):
    def manifest(self):
        rows=[]
        for template in ('T01','T02','T03','T04','T05','T06','T07','T08'):
            for depth in (1,2):
                for perturbation in (0,1):
                    for index in range(8):
                        identity=f'{template}_{depth}_{perturbation}_{index}'
                        rows.append(dict(instance_id=identity,path=identity,split='train',template_id=template,depth_level=depth,perturbation_level=perturbation))
        rows.append(dict(rows[0],instance_id='dev-secret',split='dev'))
        return dict(instances=rows)

    def test_forty_cover_templates_and_strata_without_outcome_or_order_dependence(self):
        manifest=self.manifest();selected=select_training_tasks(manifest,40)
        self.assertEqual(len(selected),40)
        for template in ('T01','T02','T03','T04','T05','T06','T07','T08'):
            rows=[row for row in selected if row['template_id']==template]
            self.assertEqual(len(rows),5)
            self.assertEqual({(row['depth_level'],row['perturbation_level']) for row in rows},{(1,0),(1,1),(2,0),(2,1)})
        altered=copy.deepcopy(manifest);altered['instances'].reverse()
        for row in altered['instances']:row['reference_grade']={'full_success':False}
        self.assertEqual([row['instance_id'] for row in selected],[row['instance_id'] for row in select_training_tasks(altered,40)])
        self.assertTrue(all(row['split']=='train' for row in selected))
        self.assertNotEqual([row['instance_id'] for row in selected],[row['instance_id'] for row in select_training_tasks(manifest,40,'different-seed')])

    def test_finite_caps_small_pools_and_duplicate_identity(self):
        manifest=self.manifest()
        for cap in (0,-1,True,None):
            with self.assertRaises(ValueError):select_training_tasks(manifest,cap)
        self.assertEqual(len(select_training_tasks(manifest,10000)),256)
        with self.assertRaises(ValueError):select_training_tasks(dict(instances=[]),40)
        manifest['instances'].append(manifest['instances'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):select_training_tasks(manifest,40)

    def test_plan_and_collector_bind_identical_selection_before_dispatch(self):
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root/'tasks',templates=['T01','T02'],count_per_template=3,split='train')
            source=root/'tasks/manifest.json'
            record=plan(source,root/'pilot',max_tasks=2)
            self.assertEqual(record['template_counts'],{'T01':1,'T02':1})
            config=dict(endpoint='http://localhost:8000/v1/chat/completions',model='fixture',model_version='fixture',tokenizer_path=str(root),tokenizer_revision='fixture',mode='full',budget=dict(tool_calls=10,generated_tokens=100,wall_seconds=20,parse_retries=2),max_context_tokens=8192,max_output_tokens=512)
            path=root/'config.json';path.write_text(json.dumps(config));seen=[]
            def execute(instance,out,frozen):
                selection=json.loads((root/'collection/selection.json').read_text())
                evidence=json.loads((root/'collection/source.json').read_text())
                self.assertEqual(selection,record)
                self.assertEqual(evidence['selection_sha256'],hashlib.sha256((root/'collection/selection.json').read_bytes()).hexdigest())
                seen.append(Path(instance).name)
                return dict(termination=dict(state='DECLARED_DONE',generated_tokens=1,input_tokens=1,tool_calls=1),runner_wall_seconds=.1,grade=dict(full_success=True))
            report=collect(source,path,root/'collection',max_attempts_per_task=3,max_episodes=6,max_generated_tokens=600,max_tasks=2,run=execute)
            self.assertEqual(seen,[row['instance_id'] for row in record['tasks']])
            self.assertEqual(report['selected_tasks'],2);self.assertEqual(report['attempted_tasks'],2)
            self.assertEqual(report['attempts'],2)
            with self.assertRaises(FileExistsError):plan(source,root/'pilot',max_tasks=2)

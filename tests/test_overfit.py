"""Overfit criteria are authored software fixtures, not learning evidence."""
import copy
import unittest
from officegym.training.overfit import validate_config,validate_rows,compare


def criteria():return dict(max_mean_nll=.1,min_token_accuracy=.99,min_relative_nll_reduction=.9,max_evaluation_wall_seconds=5)


def metric(nll,accuracy):return dict(schema='officegym-sixteen-sample-measurement-v1',samples=16,rows_sha256='0'*64,supervised_tokens=32,mean_nll=nll,token_accuracy=accuracy,sequence_exact_accuracy=accuracy)


class OverfitTests(unittest.TestCase):
    def test_predeclared_criteria_and_data_identity(self):
        report=compare(metric(2.,.2),metric(.05,1.),criteria())
        self.assertTrue(report['passed']);self.assertFalse(report['environment_success_measured']);self.assertEqual(report['independent_dev_evaluation_status'],'pending')
        self.assertFalse(compare(metric(2.,.2),metric(.05,.98),criteria())['passed'])
        self.assertFalse(compare(metric(2.,.2),metric(1.,1.),criteria())['passed'])
        changed=metric(.05,1.);changed['rows_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'data changed'):compare(metric(2.,.2),changed,criteria())
        for key,value in [('max_mean_nll',True),('min_token_accuracy',1.1),('min_relative_nll_reduction',-1),('max_evaluation_wall_seconds',float('inf'))]:
            config=criteria();config[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(config)
        invalid=metric(float('nan'),1.)
        with self.assertRaises(ValueError):compare(metric(2.,.2),invalid,criteria())

    def test_exactly_sixteen_distinct_causally_supervised_samples(self):
        rows=[dict(instance_id=f'fixture-{index}',input_ids=[0,1],assistant_mask=[0,1],token_roles=['user','assistant'],labels=[-100,1]) for index in range(16)]
        validate_rows(rows)
        with self.assertRaisesRegex(ValueError,'exactly sixteen'):validate_rows(rows[:15])
        duplicate=copy.deepcopy(rows);duplicate[-1]['instance_id']=duplicate[0]['instance_id']
        with self.assertRaisesRegex(ValueError,'distinct'):validate_rows(duplicate)
        duplicate[-1].update(input_ids=[0,2],labels=[-100,2])
        validate_rows(duplicate)  # Different trajectories of one task remain legitimate samples.
        first_only=copy.deepcopy(rows);first_only[0].update(assistant_mask=[1,0],token_roles=['assistant','tool'],labels=[0,-100])
        with self.assertRaisesRegex(ValueError,'causally predicted'):validate_rows(first_only)

    def test_real_sft_preflight_requires_fixed_sixteen_and_explicit_criteria(self):
        import json,tempfile
        from pathlib import Path
        from review_fixtures import reviewed_fixture
        from officegym.training.sft import validate_plan
        from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
        from officegym.harness.versions import HARNESS_VERSION
        for count in (5,16):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);data=root/'data'
                reviewed_fixture(data,dict(harness_version=HARNESS_VERSION,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION),count=count)
                plan=dict(model_path=directory,model_revision='authored-fixture',max_steps=1,max_wall_seconds=60,checkpoint_interval_steps=1,locked_packages={name:'fixture' for name in ('torch','transformers','trl','peft','datasets')},max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=.0001,lora={},overfit_check=criteria())
                if count==5:
                    with self.assertRaisesRegex(ValueError,'exactly sixteen'):validate_plan(plan,data)
                else:self.assertEqual(len(validate_plan(plan,data)['accepted']),16)
                plan['overfit_check']['max_mean_nll']='SET_BY_OPERATOR'
                with self.assertRaisesRegex(ValueError,'predeclared'):validate_plan(plan,data)

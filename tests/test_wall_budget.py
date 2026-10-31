import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from officegym.harness.loop import Harness,Budget,AssistantTurn


class FinishPolicy:
    model_version='authored-latency-fixture'
    def __init__(self):
        self.calls=0
    def generate(self,*args):
        self.calls+=1
        return AssistantTurn('task.finish',{'summary':'fixture'},5,3)


class WallBudgetTests(unittest.TestCase):
    def test_startup_origin_exhausts_budget_before_model_request(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=FinishPolicy()
            harness=Harness(policy,lambda *args:self.fail('tool must not execute'),{},Path(directory)/'trace.jsonl',Budget(wall_seconds=.1))
            result=harness.run(started_at=time.monotonic()-.2)
            self.assertEqual(result['state'],'BUDGET_STOP')
            self.assertEqual(policy.calls,0)
            self.assertGreaterEqual(result['wall_seconds'],.2)
            self.assertEqual(result['generated_tokens'],0)

    def test_late_finish_keeps_observation_but_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            effects=[]
            def call(*args):
                effects.append('committed finish fixture')
                time.sleep(.15)
                return {'ok':True,'data':{}}
            trace=Path(directory)/'trace.jsonl'
            result=Harness(FinishPolicy(),call,{},trace,Budget(wall_seconds=.1)).run()
            self.assertEqual(result['state'],'BUDGET_STOP')
            self.assertEqual(effects,['committed finish fixture'])
            self.assertEqual(result['tool_calls'],1)
            self.assertEqual(result['generated_tokens'],3)
            records=[json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len([r for r in records if r['kind']=='interaction']),1)
            self.assertGreaterEqual(result['wall_seconds'],.15)

    def test_timeout_is_budget_stop_only_after_actual_wall_exhaustion(self):
        for delayed in (False,True):
            with self.subTest(delayed=delayed),tempfile.TemporaryDirectory() as directory:
                def call(*args):
                    if delayed:
                        time.sleep(.15)
                    raise TimeoutError('authored transport timeout')
                result=Harness(FinishPolicy(),call,{},Path(directory)/'trace.jsonl',Budget(wall_seconds=.1)).run()
                self.assertEqual(result['state'],'BUDGET_STOP' if delayed else 'INFRA_ERROR')
                self.assertEqual(result['tool_calls'],1)

    def test_invalid_future_and_nonfinite_origins_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            harness=Harness(FinishPolicy(),lambda *args:None,{},Path(directory)/'trace.jsonl')
            for origin in (float('nan'),float('inf'),time.monotonic()+100,True):
                with self.subTest(origin=origin),self.assertRaises(ValueError):
                    harness.run(started_at=origin)

    def test_episode_startup_timeout_stays_budget_failure_without_policy(self):
        from officegym.eval.episode import run_episode
        from officegym.tasks.generate import generate
        from officegym.harness.policy import ChatPolicy
        from officegym.harness.isolation import require_isolation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            transports=[]
            policy=ChatPolicy('fixture','fixture','fixture-version',transport=lambda *args:transports.append(args))
            startup_caps=[]
            def delayed_startup(*args,**kwargs):
                startup_caps.append(kwargs['startup_seconds'])
                time.sleep(.35)
                raise TimeoutError('authored late isolation startup')
            with patch('officegym.harness.isolation.IsolatedPolicy',side_effect=delayed_startup):
                result=run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',policy,budget=Budget(wall_seconds=.3))
            self.assertEqual(result['termination']['state'],'BUDGET_STOP')
            self.assertEqual(transports,[])
            self.assertIsNone(result['infrastructure_error'])
            self.assertFalse(result['grade']['full_success'])
            self.assertEqual(result['termination']['generated_tokens'],0)
            self.assertEqual(result['policy_isolation']['status'],'budget_exhausted')
            self.assertFalse(require_isolation(root/'run',result,allow_startup_failure=True))
            self.assertTrue(startup_caps)
            self.assertTrue(0<startup_caps[0]<=.3)

    def test_world_rpc_timeout_is_capped_by_remaining_episode_budget(self):
        from officegym.eval.episode import run_episode
        from officegym.env.service import Client
        from officegym.tasks.generate import generate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            timeouts=[]
            def client(path,timeout):
                timeouts.append(timeout)
                return Client(path,timeout=timeout)
            with patch('officegym.eval.episode.Client',side_effect=client):
                result=run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',FinishPolicy(),budget=Budget(wall_seconds=2))
            self.assertEqual(result['termination']['state'],'DECLARED_DONE')
            self.assertTrue(timeouts)
            self.assertTrue(all(0<value<2 for value in timeouts))

    def test_recovery_downtime_exhausts_budget_without_new_request(self):
        class FailedPolicy(FinishPolicy):
            def generate(self,*args):
                self.calls+=1
                raise ConnectionError('authored interruption')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.jsonl'
            budget=Budget(wall_seconds=.1)
            first=Harness(FailedPolicy(),None,{},path,budget)
            self.assertEqual(first.run()['state'],'INFRA_ERROR')
            time.sleep(.15)
            policy=FinishPolicy()
            restored=Harness(policy,None,{},path,budget)
            usage=restored.restore()
            self.assertGreaterEqual(usage['wall_seconds'],.15)
            self.assertEqual(restored.run()['state'],'BUDGET_STOP')
            self.assertEqual(policy.calls,0)
            with self.assertRaisesRegex(ValueError,'frozen wall'):
                restored.run(started_at=time.monotonic())

    def test_recovery_rejects_clock_identity_changes(self):
        class FailedPolicy(FinishPolicy):
            def generate(self,*args):
                raise ConnectionError('authored interruption')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.jsonl'
            Harness(FailedPolicy(),None,{},path).run()
            with patch('officegym.harness.loop.wall_clock_identity',return_value='other-boot-or-clock'):
                with self.assertRaisesRegex(ValueError,'clock identity'):
                    Harness(FailedPolicy(),None,{},path).restore()

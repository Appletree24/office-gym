"""Shared-boundary recovery uses real tool worlds and authored policies only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.env.world import World
from officegym.eval.fault_probe import run
from officegym.eval.fault_checkpoint import verify_checkpoint,branch_checkpoint,verify_branch
from officegym.harness.loop import Harness,AssistantTurn,Budget
from officegym.harness.journal import Journal
from officegym.tools.dispatcher import Dispatcher

TASK={'goal':'Send review invitation'}


class Policy:
    def __init__(self,mode):self.mode=mode;self.calls=0
    def generate(self,messages,*unused):
        self.calls+=1
        if self.mode=='simple' and self.calls==1:
            notices=[json.loads(m['content']) for m in messages if m['role']=='user' and 'controller_recovery_notice' in m['content']]
            operation=notices[-1]['pending_actions'][0]['operation_id']
            return AssistantTurn('operations.get',{'operation_id':operation},3,2)
        return AssistantTurn('task.finish',{'summary':'Authored shared recovery'},6,4)


class FaultRestoreTests(unittest.TestCase):
    def test_both_methods_inherit_same_prefix_and_simple_queries_only_by_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run(root/'probe');source=root/'probe/committed_checkpoint';record=verify_checkpoint(source);results={};origins=[]
            for mode in ('simple','full'):
                branch=root/mode;branch_checkpoint(source,branch);world=World(branch/'world.sqlite');journal=Journal(branch/'runtime.sqlite') if mode=='full' else None;client=Dispatcher(world);calls=[];policy=Policy(mode)
                def call(tool,args,operation=None):calls.append(tool);return client.call(tool,args,operation)
                harness=Harness(policy,call,TASK,branch/'trajectory.jsonl',journal=journal)
                try:
                    metadata=harness.restore_fault_checkpoint(source);origins.append(harness.wall_origin)
                    self.assertEqual(metadata['prior_usage']['generated_tokens'],8);self.assertEqual(metadata['prior_usage']['input_tokens'],10);self.assertFalse(metadata['clock_reset']);self.assertEqual(calls,[])
                    if mode=='simple':
                        self.assertEqual(harness.pending_actions,{})
                        self.assertEqual(harness.messages[-1]['role'],'user');self.assertEqual(len([m for m in harness.messages if m['role']=='tool']),0)
                    else:harness.reconcile_pending();self.assertEqual(calls,['operations.get'])
                    termination=harness.run();self.assertEqual(termination['state'],'DECLARED_DONE');self.assertEqual(termination['tool_calls'],3)
                    self.assertEqual(calls,['operations.get','task.finish']);self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
                    self.assertEqual(termination['generated_tokens'],14 if mode=='simple' else 12)
                    self.assertEqual(policy.calls,2 if mode=='simple' else 1)
                    records=[json.loads(line) for line in (branch/'trajectory.jsonl').read_text().splitlines()]
                    self.assertTrue(records[0]['fault_branch']);self.assertEqual(records[0]['mode'],mode);self.assertEqual(records[0]['recovery']['source_checkpoint_sha256'],metadata['source_checkpoint_sha256'])
                    fresh=Harness(Policy(mode),call,TASK,branch/'trajectory.jsonl',journal=journal)
                    with self.assertRaisesRegex(ValueError,'ordinary restore is forbidden'):fresh.restore()
                    results[mode]=metadata['source_checkpoint_sha256']
                finally:
                    if journal:journal.close()
                    world.close()
            self.assertEqual(origins[0],origins[1]);self.assertEqual(results['simple'],results['full']);self.assertEqual(verify_checkpoint(source),record)

    def test_expired_source_clock_cannot_grant_new_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run(root/'probe');source=root/'probe/committed_checkpoint';branch=root/'simple';branch_checkpoint(source,branch)
            header=json.loads((branch/'harness.jsonl').read_text().splitlines()[0]);origin=header['wall_clock']['monotonic_origin'];policy=Policy('simple');harness=Harness(policy,lambda *args:self.fail('expired budget called tool'),TASK,branch/'trajectory.jsonl')
            with patch('officegym.harness.loop.time.monotonic',return_value=origin+601):
                metadata=harness.restore_fault_checkpoint(source);termination=harness.run()
            self.assertGreaterEqual(metadata['prior_usage']['wall_seconds'],601);self.assertEqual(termination['state'],'BUDGET_STOP');self.assertEqual(policy.calls,0);verify_checkpoint(source)

    def test_changed_model_budget_and_branch_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run(root/'probe');source=root/'probe/committed_checkpoint'
            for fault in ('model','budget','branch'):
                branch=root/fault;branch_checkpoint(source,branch);policy=Policy('simple');budget=Budget()
                if fault=='model':policy.model_version='different'
                if fault=='budget':budget=Budget(tool_calls=199)
                if fault=='branch':
                    with (branch/'harness.jsonl').open('a') as stream:stream.write('\n')
                harness=Harness(policy,lambda *args:self.fail('invalid branch called tool'),TASK,branch/'trajectory.jsonl',budget)
                with self.subTest(fault=fault),self.assertRaises(ValueError):harness.restore_fault_checkpoint(source)
                self.assertFalse((branch/'trajectory.jsonl').exists())
            verify_checkpoint(source)

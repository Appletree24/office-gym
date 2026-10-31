"""Trusted rollout lifecycle bridge for a token-native training backend.

This module is framework independent; a locked verl BaseTool/AgentLoop wrapper
must invoke it without exposing reward/private spec to the actor.
"""
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tools.schema import WRITES
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.eval.artifacts import write_json
from .data import validate_rollout
from officegym.tasks.partitions import validate_partition


class RolloutEnvironment:
    def __init__(self,instance_dir,*,actor_version,max_tool_calls=24,work_dir=None,artifact_dir=None):
        if not actor_version or type(max_tool_calls) is not int or max_tool_calls<1:
            raise ValueError('locked actor and finite budget required')
        source = Path(instance_dir).resolve()
        task = json.loads((source/'task.json').read_text())
        if task['split']!='train':
            raise ValueError('training rollout requires train split')
        validate_partition(task['template_id'],task['split'],task.get('seed'))
        self.public_task = task['public_task']
        self._spec = {**self.public_task,**json.loads((source/'private_spec.json').read_text())}
        self.actor_version = actor_version
        self._initial = source/'initial_world.sqlite'
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else None
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True,exist_ok=False)
            (self.artifact_dir/'provenance.json').write_text(canonical({'instance_id':task['instance_id'],'actor_version':actor_version,'source_instance':str(source)})+'\n')
        self._temporary = tempfile.TemporaryDirectory(prefix='officegym_rl_',dir=work_dir)
        self._path = Path(self._temporary.name)/'world.sqlite'
        self._world = clone_world(self._initial,self._path)
        try:
            with self._world.db:
                calls = json.loads(self._world.db.execute("SELECT value_json FROM meta WHERE key='tool_calls'").fetchone()[0])
                if calls!=0:
                    raise ValueError('rollout snapshot already executed')
                self._world.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('max_tool_calls',canonical(max_tool_calls)))
            self._client = Dispatcher(self._world,max_calls=max_tool_calls)
        except BaseException:
            self.close()
            raise
        self._actions = 0
        self._graded = False
        self.closed = False

    def call(self,name,arguments):
        if self.closed or self._graded:
            raise ValueError('rollout no longer accepts actions')
        self._actions += 1
        operation = 'actor_write_'+str(self._actions) if name in WRITES else None
        result = self._client.call(name,arguments,operation)
        if self.artifact_dir:
            with (self.artifact_dir/'tools.jsonl').open('a') as stream:
                stream.write(canonical({'tool':name,'arguments':arguments,'operation_id':operation,'result':result})+'\n')
                stream.flush()
                os.fsync(stream.fileno())
        return result

    def finalize(self,*,input_ids,response_mask,token_roles,logprobs,actor_version,termination_reason='policy_terminated'):
        if self.closed or self._graded or actor_version!=self.actor_version:
            raise ValueError('rollout lifecycle or actor version mismatch')
        if termination_reason not in ('policy_terminated','response_limit','generation_budget','invalid_tool_batch'):
            raise ValueError('unknown rollout termination reason')
        validate_rollout(input_ids,response_mask,token_roles,logprobs)
        domain_grade = evaluate(self._path,self._spec,self._initial)
        audit = reconcile(self._path)
        grade = dict(domain_grade)
        if termination_reason!='policy_terminated' or not audit['consistent']:
            failures = list(grade['failures'])
            if termination_reason!='policy_terminated':
                failures.append('rollout_'+termination_reason)
            if not audit['consistent']:
                failures.append('operation_audit_inconsistent')
            grade.update(full_success=False,reward=-1.0 if grade['hard_violations'] else 0.0,failures=failures)
        self._graded = True
        result = {'reward':grade['reward'],'grade':grade,'domain_grade':domain_grade,'termination_reason':termination_reason,'operation_audit':audit,'final_state_hash':self._world.state_hash(),
                'actor_version':actor_version,'tool_calls':self._client.meta('tool_calls'),'original_token_ids':list(input_ids),'response_mask':list(response_mask),'token_roles':list(token_roles),'original_logprobs':list(logprobs)}
        if self.artifact_dir:
            destination = sqlite3.connect(self.artifact_dir/'final_world.sqlite')
            try:
                self._world.db.backup(destination)
            finally:
                destination.close()
            write_json(self.artifact_dir/'terminal.json',result)
        return result

    def close(self):
        if getattr(self,'closed',False):
            return
        try:
            if hasattr(self,'_world'):
                archive = getattr(self,'artifact_dir',None)
                if archive and not (archive/'final_world.sqlite').exists():
                    destination = sqlite3.connect(archive/'final_world.sqlite')
                    try:
                        self._world.db.backup(destination)
                    finally:
                        destination.close()
                if archive and not (archive/'terminal.json').exists():
                    write_json(archive/'incomplete.json',{'terminal_grade_available':False})
        finally:
            try:
                if hasattr(self,'_world'):
                    self._world.close()
            finally:
                try:
                    self._temporary.cleanup()
                finally:
                    self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()

"""Trusted single-episode runner. Retains failed episodes and raw model traces."""
import json
import multiprocessing
import time
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.env.service import serve,Client
from officegym.harness.loop import Harness,Budget
from officegym.harness.journal import Journal
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from .artifacts import write_json


class EpisodeWallBudgetExhausted(TimeoutError):
    pass


def run_episode(instance_dir,out,policy,*,mode='full',budget=None,context_manager=None,initial_metadata=None):
    if mode not in ('simple','full'):
        raise ValueError('unknown harness mode')
    started=time.monotonic()
    initial_metadata = initial_metadata or {}
    if not isinstance(initial_metadata,dict) or set(initial_metadata)-{'resolved_eval_config.json','tokenizer_manifest.json'}:
        raise ValueError('unsupported initial episode metadata')
    for value in initial_metadata.values():
        canonical(value)
    instance_dir,out = Path(instance_dir).resolve(),Path(out).resolve()
    task = json.loads((instance_dir/'task.json').read_text())
    public = task['public_task']
    private = json.loads((instance_dir/'private_spec.json').read_text())
    budget = budget or Budget()
    out.mkdir(parents=True,exist_ok=False)
    for name,value in initial_metadata.items():
        write_json(out/name,value)
    from .provenance import provenance
    versions = provenance(instance_dir,policy)
    write_json(out/'provenance.json',versions)
    world_path = out/'world.sqlite'
    world = clone_world(instance_dir/'initial_world.sqlite',world_path)
    try:
        meta = {row['key']:json.loads(row['value_json']) for row in world.db.execute('SELECT * FROM meta')}
        if meta.get('tool_calls',0)!=0 or meta.get('finished') or world.db.execute('SELECT 1 FROM operations LIMIT 1').fetchone():
            raise ValueError('evaluation requires an unexecuted initial snapshot')
        with world.db:
            world.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('max_tool_calls',canonical(budget.tool_calls)))
    finally:
        world.close()
    from dataclasses import asdict
    write_json(out/'budget.json',asdict(budget))
    socket_path = out/'world.sock'
    process = multiprocessing.get_context('spawn').Process(target=serve,args=(world_path,socket_path,budget.tool_calls))
    journal = None
    isolated_policy = None
    original_policy = policy
    context_policy_replaced = False
    termination = {'state':'INFRA_ERROR'}
    error = None
    deadline=started+budget.wall_seconds
    def remaining_wall():
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise EpisodeWallBudgetExhausted('episode execution wall budget exhausted')
        return remaining
    def bounded_call(name,arguments,operation_id=None):
        return Client(socket_path,timeout=min(30,remaining_wall())).call(name,arguments,operation_id)
    from officegym.harness.policy import ChatPolicy
    requires_isolation=isinstance(policy,ChatPolicy) or getattr(policy,'policy_kind',None)=='inference_service'
    isolation_evidence=dict(version='policy-chroot-v1',verified=False,status='not_started') if requires_isolation else dict(version='software_fixture_not_isolated',verified=False)
    write_json(out/'policy_isolation.json',isolation_evidence)
    try:
        remaining_wall()
        process.start()
        while not socket_path.exists():
            remaining_wall()
            if not process.is_alive() or time.monotonic()-started>10:
                raise RuntimeError('world service startup failed')
            time.sleep(min(.01,remaining_wall()))
        remaining_wall()
        if requires_isolation:
            from officegym.harness.isolation import IsolatedPolicy
            isolated_policy=IsolatedPolicy(policy,protected_paths=[instance_dir/'private_spec.json',instance_dir/'event_rules.json',instance_dir/'initial_world.sqlite',world_path,Path(__file__).parents[1]/'graders'/'result.py'],startup_seconds=min(10,remaining_wall()))
            isolation_evidence=isolated_policy.evidence
            write_json(out/'policy_isolation.json',isolation_evidence)
            if context_manager is not None and hasattr(context_manager,'policy'):
                if context_manager.policy is not policy:
                    raise ValueError('compression must use the executing policy')
                context_manager.policy=isolated_policy
                context_policy_replaced=True
            policy=isolated_policy
        remaining_wall()
        journal = Journal(out/'runtime.sqlite') if mode=='full' else None
        harness = Harness(policy,bounded_call,public,out/'trajectory.jsonl',budget,journal,context_manager)
        termination = harness.run(started_at=started)
    except Exception as exc:
        budget_stop=isinstance(exc,EpisodeWallBudgetExhausted) or isinstance(exc,TimeoutError) and time.monotonic()>=deadline
        if budget_stop:
            termination=dict(state='BUDGET_STOP',tool_calls=0,generated_tokens=0,input_tokens=0,wall_seconds=time.monotonic()-started)
        else:
            error = {'type':type(exc).__name__,'message':str(exc)}
        if not isolation_evidence.get('verified') and requires_isolation:
            isolation_evidence.update(status='budget_exhausted' if budget_stop else 'failed',error_type=type(exc).__name__)
            write_json(out/'policy_isolation.json',isolation_evidence)
        if not (out/'trajectory.jsonl').exists():
            (out/'trajectory.jsonl').write_text(canonical(dict(kind='wall_budget_exhausted_before_policy' if budget_stop else 'infrastructure_failure_before_policy',error_type=type(exc).__name__))+'\n')
    finally:
        if context_policy_replaced:
            context_manager.policy=original_policy
        if isolated_policy:
            isolated_policy.close()
        if journal:
            journal.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join()
    grade = evaluate(world_path,{**public,**private},instance_dir/'initial_world.sqlite')
    audit = reconcile(world_path)
    if termination['state'] in ('INFRA_ERROR','PARSE_ERROR','BUDGET_STOP','CONTEXT_OVERFLOW') or error or not audit['consistent']:
        grade['full_success'] = False
        grade['reward'] = -1.0 if grade['hard_violations'] else 0.0
    result = dict(instance_id=task['instance_id'],split=task['split'],template_id=task['template_id'],mode=mode,
                  model_version=getattr(policy,'model_version',None),policy_kind=getattr(policy,'policy_kind','software_fixture'),termination=termination,grade=grade,operation_audit=audit,
                  policy_isolation=isolation_evidence,infrastructure_error=error,runner_wall_seconds=time.monotonic()-started,
                  scope='single_episode_not_complete_evaluation_matrix',versions=versions)
    write_json(out/'result.json',result)
    return result

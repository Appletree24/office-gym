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


def run_episode(instance_dir,out,policy,*,mode='full',budget=None,context_manager=None,initial_metadata=None):
    if mode not in ('simple','full'):
        raise ValueError('unknown harness mode')
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
    termination = {'state':'INFRA_ERROR'}
    started = time.monotonic()
    error = None
    try:
        process.start()
        while not socket_path.exists():
            if not process.is_alive() or time.monotonic()-started>10:
                raise RuntimeError('world service startup failed')
            time.sleep(.01)
        journal = Journal(out/'runtime.sqlite') if mode=='full' else None
        harness = Harness(policy,Client(socket_path).call,public,out/'trajectory.jsonl',budget,journal,context_manager)
        termination = harness.run()
    except Exception as exc:
        error = {'type':type(exc).__name__,'message':str(exc)}
    finally:
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
                  infrastructure_error=error,runner_wall_seconds=time.monotonic()-started,
                  scope='single_episode_not_complete_evaluation_matrix',versions=versions)
    write_json(out/'result.json',result)
    return result

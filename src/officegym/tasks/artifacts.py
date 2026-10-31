"""Trusted task artifact writer; reference receives public inputs only."""
import hashlib
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.tools.dispatcher import Dispatcher
from .reference import solve
from .integrity import fingerprints,dependency_depth
from .validate import REQUIRED


def write_instance(root, task, initial_world, private_spec, event_rules, identity_map=None):
    root = Path(root)
    identity = task['instance_id']
    if not isinstance(identity,str) or not identity or Path(identity).name != identity or identity in ('.','..'):
        raise ValueError('invalid instance identity')
    directory = root/identity
    directory.mkdir(parents=True,exist_ok=False)
    dag = task['dependency_dag']
    task = dict(task,dependency_path_length=dependency_depth(dag['nodes'],dag['edges']))
    public = task['public_task']
    fp = fingerprints(public,initial_world.snapshot(),event_rules,'env-v1',identity_map=identity_map)
    snapshot = directory/'initial_world.sqlite'
    import sqlite3
    destination = sqlite3.connect(snapshot)
    try:
        initial_world.db.backup(destination)
    finally:
        destination.close()
    for filename,content in [('task.json',task),('private_spec.json',private_spec),('event_rules.json',event_rules)]:
        (directory/filename).write_text(canonical(content)+'\n')
    final_path = directory/'reference_final.sqlite'
    world = clone_world(snapshot,final_path)
    try:
        client = Dispatcher(world)
        with (directory/'reference_trace.jsonl').open('w') as trace:
            def call(name,args,operation=None):
                result = client.call(name,args,operation)
                trace.write(canonical(dict(tool=name,arguments=args,operation_id=operation,result=result))+'\n')
                trace.flush()
                return result
            solve(public,call)
        grade = evaluate(final_path,{**public,**private_spec},snapshot)
        (directory/'reference_grade.json').write_text(canonical(grade)+'\n')
        audit = reconcile(final_path)
        (directory/'reference_operations.json').write_text(canonical(audit)+'\n')
        if not audit['consistent']:
            raise ValueError('reference operation reconciliation failed: '+canonical(audit))
        if not grade['full_success']:
            raise ValueError('generated task reference failed: '+canonical(grade))
        terminal_hash = world.state_hash()
    finally:
        world.close()
    entry = {key:task[key] for key in ('instance_id','split','template_id','scenario_family_id','depth_level','perturbation_level')}
    entry.update(fp,path=identity,reference_final_hash=terminal_hash,identity_map=identity_map or {},env_version='env-v1',
                 files_sha256={filename:hashlib.sha256((directory/filename).read_bytes()).hexdigest() for filename in REQUIRED})
    return entry

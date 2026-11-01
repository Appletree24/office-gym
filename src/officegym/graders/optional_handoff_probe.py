"""Test a public-directory handoff shortcut; no private facts or fake observations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve,PUBLIC_HANDOFF_VARIANT
from officegym.tasks.validate import validate_dataset
from .result import evaluate


def run(manifest_path,out,*,max_instances,max_wall_seconds):
    if type(max_instances) is not int or max_instances<1 or type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:raise ValueError('finite handoff probe budgets required')
    started=time.monotonic();source=Path(manifest_path).resolve();source_sha=hashlib.sha256(source.read_bytes()).hexdigest();validate_dataset(source);manifest=json.loads(source.read_text());entries=[]
    for entry in manifest['instances']:
        task=json.loads((source.parent/entry['path']/'task.json').read_text())
        if task['public_task'].get('handoff_thread_id'):entries.append((entry,task))
    if not entries or len(entries)>max_instances:raise ValueError('instance cap must cover all handoff tasks')
    root=Path(out);root.mkdir(parents=True,exist_ok=False);records=[]
    with (root/'cases.jsonl').open('x') as stream:
        for entry,task in entries:
            if time.monotonic()-started>=max_wall_seconds:raise TimeoutError('handoff probe wall budget exhausted')
            directory=root/entry['instance_id'];directory.mkdir();original=source.parent/entry['path'];world=clone_world(original/'initial_world.sqlite',directory/'final.sqlite');client=Dispatcher(world);calls=0
            try:
                with (directory/'actions.jsonl').open('x') as trace:
                    def call(name,args,key=None):
                        nonlocal calls
                        if name=='mail.get_thread' and args['thread_id']==task['public_task']['handoff_thread_id']:raise ValueError('shortcut read the handoff thread')
                        calls+=1;result=client.call(name,args,key);trace.write(canonical(dict(tool=name,arguments=args,operation_id=key,result=result))+'\n');return result
                    solve(task['public_task'],call,handoff_observations=False)
                spec={**task['public_task'],**json.loads((original/'private_spec.json').read_text())};grade=evaluate(directory/'final.sqlite',spec,original/'initial_world.sqlite');(directory/'grade.json').write_text(canonical(grade)+'\n')
                record=dict(instance_id=entry['instance_id'],declared_handoff_node='read_handoff_chain' in task['dependency_dag']['nodes'],full_success=grade['full_success'],within_budget=grade['within_budget'],declared_done=grade['declared_done'],actual_handoff_reads=0,real_tool_calls=calls)
                records.append(record);stream.write(canonical(record)+'\n');stream.flush()
            finally:world.close()
    report=dict(kind='authored-public-directory-handoff-shortcut-not-model-results',reference_variant=PUBLIC_HANDOFF_VARIANT,source_manifest_sha256=source_sha,instances=len(records),declared_handoff_nodes=sum(row['declared_handoff_node'] for row in records),successful=sum(row['full_success'] for row in records),all_shortcuts_legal=all(row['full_success'] and row['within_budget'] and row['declared_done'] for row in records),actual_handoff_reads=0,elapsed_seconds=time.monotonic()-started,model_calls=0,scope='Current unique-other-person fixtures only; not a general handoff-resolution algorithm.')
    (root/'summary.json').write_text(canonical(report)+'\n')
    if not report['all_shortcuts_legal']:raise ValueError('public handoff shortcut failed; inspect retained cases')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True);parser.add_argument('--max-instances',type=int,required=True);parser.add_argument('--max-wall-seconds',type=float,required=True);args=parser.parse_args();print(json.dumps(run(args.manifest,args.out,max_instances=args.max_instances,max_wall_seconds=args.max_wall_seconds),indent=2))

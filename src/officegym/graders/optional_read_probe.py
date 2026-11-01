"""Test public material binding with documents.read omitted; authored, not a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve,PUBLIC_MATERIAL_VARIANT
from officegym.tasks.validate import validate_dataset


def run(manifest_path,out,*,max_instances,max_wall_seconds):
    if type(max_instances) is not int or max_instances<1 or type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:raise ValueError('finite optional-read probe budgets required')
    started=time.monotonic();source=Path(manifest_path).resolve();source_sha=hashlib.sha256(source.read_bytes()).hexdigest();validate_dataset(source)
    manifest=json.loads(source.read_text());entries=[]
    for entry in manifest['instances']:
        task=json.loads((source.parent/entry['path']/'task.json').read_text())
        if task['public_task'].get('required_document_refs'):entries.append((entry,task))
    if not entries or len(entries)>max_instances:raise ValueError('instance cap must cover all material-binding tasks')
    root=Path(out);root.mkdir(parents=True,exist_ok=False);records=[]
    with (root/'cases.jsonl').open('x') as stream:
        for entry,task in entries:
            if time.monotonic()-started>=max_wall_seconds:raise TimeoutError('optional-read probe wall budget exhausted')
            directory=root/entry['instance_id'];directory.mkdir();original=source.parent/entry['path'];world=clone_world(original/'initial_world.sqlite',directory/'final.sqlite');client=Dispatcher(world);baseline_reads=sum(json.loads(line).get('tool')=='documents.read' for line in (original/'reference_trace.jsonl').read_text().splitlines() if line.strip())
            try:
                with (directory/'actions.jsonl').open('x') as trace:
                    def call(name,args,key=None):
                        if name=='documents.read':raise ValueError('public-reference-only variant unexpectedly requested a material read')
                        result=client.call(name,args,key)
                        trace.write(canonical(dict(tool=name,arguments=args,operation_id=key,executed=True,result=result))+'\n');return result
                    solve(task['public_task'],call,material_observations=False)
                from .result import evaluate
                spec={**task['public_task'],**json.loads((original/'private_spec.json').read_text())}
                grade=evaluate(directory/'final.sqlite',spec,original/'initial_world.sqlite');(directory/'grade.json').write_text(canonical(grade)+'\n')
                record=dict(instance_id=entry['instance_id'],split=entry['split'],template_id=entry['template_id'],declared_read_node='read_material' in task['dependency_dag']['nodes'],skipped_reads=baseline_reads,actual_material_reads=0,reference_variant=PUBLIC_MATERIAL_VARIANT,real_tool_calls=client.meta('tool_calls'),full_success=grade['full_success'],within_budget=grade['within_budget'],declared_done=grade['declared_done'])
                records.append(record);stream.write(canonical(record)+'\n');stream.flush()
            finally:world.close()
    report=dict(kind='authored-public-material-binding-without-read-not-model-results',source_manifest_sha256=source_sha,instances=len(records),declared_read_nodes=sum(row['declared_read_node'] for row in records),skipped_reads=sum(row['skipped_reads'] for row in records),skipped_reads_scope='baseline reference read calls omitted, not attempted calls',actual_material_reads=0,reference_variant=PUBLIC_MATERIAL_VARIANT,all_calls_use_real_dispatcher=True,successful=sum(row['full_success'] for row in records),all_read_omissions_legal=all(row['skipped_reads'] and row['full_success'] and row['within_budget'] and row['declared_done'] for row in records),elapsed_seconds=time.monotonic()-started,max_wall_seconds=max_wall_seconds,model_calls=0)
    (root/'summary.json').write_text(canonical(report)+'\n')
    if not report['all_read_omissions_legal']:raise ValueError('read omission failed; inspect retained counterexamples')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True);parser.add_argument('--max-instances',type=int,required=True);parser.add_argument('--max-wall-seconds',type=float,required=True);args=parser.parse_args();print(json.dumps(run(args.manifest,args.out,max_instances=args.max_instances,max_wall_seconds=args.max_wall_seconds),indent=2))

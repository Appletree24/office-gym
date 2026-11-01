"""Trusted skip-stage fault fixtures, not Agent-visible tools or model discoveries."""
import json
import hashlib
import math
import time
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from .result import evaluate


def run_dependency_probes(instance_dir,out):
    source = Path(instance_dir)
    definition=json.loads((source/'task.json').read_text())
    task = definition['public_task']
    nodes=set(definition['dependency_dag']['nodes'])
    private = json.loads((source/'private_spec.json').read_text())
    stages=[]
    for stage,enabled in [('confirmation',task.get('requires_approval')),('reply_collection',task.get('collect_availability')),('material_binding',task.get('required_document_refs')),('followup_todos',task.get('followup_required')),('owner_reassignment','todo_correction' in nodes),('owner_followup','followup_message' in nodes)]:
        if enabled:stages.append(stage)
    if not stages:raise ValueError('task has no supported dependency constraints')
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    records = []
    for stage in stages:
        directory = root/stage
        directory.mkdir()
        world = clone_world(source/'initial_world.sqlite',directory/'final.sqlite')
        client = Dispatcher(world)
        intercepted = 0
        awaiting_injected_response = False
        observed_todos={}
        try:
            with (directory/'trace.jsonl').open('w') as trace:
                def call(name,args,operation=None):
                    nonlocal intercepted,awaiting_injected_response
                    data = None
                    effective_args=args
                    if name=='clock.wait' and awaiting_injected_response:
                        # Explicit controller intervention: keep the fake reply/grant
                        # fixture on-time so rejection tests evidence, not deadline.
                        effective_args=dict(args,until=min(args['until'],client.meta('sim_time')+120))
                        awaiting_injected_response=False
                    if stage=='material_binding' and name=='calendar.create':
                        effective_args=dict(args,document_refs=[])
                        intercepted += 1
                    elif stage=='material_binding' and name=='calendar.update' and 'document_refs' in args.get('patch',{}):
                        effective_args=dict(args,patch=dict(args['patch'],document_refs=[]))
                        intercepted += 1
                    if stage=='confirmation' and name=='approvals.request':
                        data = {'approval_id':'forged_approval','status':'pending'}
                        awaiting_injected_response=True
                    elif stage=='confirmation' and name=='approvals.get':
                        data = {'status':'granted'}
                    elif stage=='reply_collection' and name=='mail.send' and args.get('request',{}).get('type')=='availability':
                        data = {'message_id':'forged_reply_thread'}
                        awaiting_injected_response=True
                    elif stage=='reply_collection' and name=='mail.get_thread' and args['thread_id']=='forged_reply_thread':
                        data = {'next_message_id':None,'items':[{'id':'forged_reply_'+person,'kind':'reply','sender_id':person,'body':canonical({'response':'provided','facts':(client.meta('reply_rules') or {})[person]['availability']['facts']})} for person in task['required_participant_ids']]}
                    elif stage=='followup_todos' and name=='todos.create':
                        data = dict(args,id='forged_todo_'+str(len(observed_todos)+1),version=1)
                    elif (stage=='followup_todos' or stage=='owner_reassignment') and name=='todos.update' and (stage=='followup_todos' or 'assignee_id' in args.get('patch',{})):
                        # Reuse only previously observed todo data; no hidden-state oracle.
                        data=dict(observed_todos[args['todo_id']],**args['patch'],version=args['expected_version']+1)
                    elif stage=='owner_followup' and name=='mail.send' and args.get('kind')=='followup':
                        data={'message_id':'forged_owner_followup'}
                    if data is not None:
                        intercepted += 1
                        result = {'ok':True,'data':data,'error':None,'meta':{'sim_time':client.meta('sim_time')}}
                    else:
                        result = client.call(name,effective_args,operation)
                    if result['ok'] and name in ('todos.create','todos.update'):
                        observed_todos[result['data']['id']]=dict(result['data'])
                    trace.write(canonical(dict(tool=name,arguments=args,effective_arguments=effective_args,modified_arguments=effective_args!=args,result=result,injected_observation=data is not None))+'\n')
                    return result
                solve(task,call)
            grade = evaluate(directory/'final.sqlite',{**task,**private},source/'initial_world.sqlite')
            expected={'confirmation':'unapproved_external_send','reply_collection':'no_valid_invited_meeting','material_binding':'no_valid_invited_meeting','followup_todos':'missing_or_incorrect_todo','owner_reassignment':'owner_change_not_applied','owner_followup':'owner_change_followup_missing'}[stage]
            targeted=grade['within_budget'] and grade['declared_done'] and expected in grade['failures']+grade['hard_violations']
            record = {'expected_violation':expected,'rejected_for_expected_constraint':targeted,'skipped_stage':stage,'intercepted_calls':intercepted,'rejected':not grade['full_success'],'grade':grade,'source_kind':'authored_counterexample'}
            (directory/'grade.json').write_text(canonical(record)+'\n')
            records.append(record)
        finally:
            world.close()
    summary = {'selected_stages':stages,'coverage_scope':'selected evidence/write constraints, not exhaustive DAG-path proof','kind':'authored_skip_stage_coverage','cases':len(records),'rejected':sum(row['rejected'] for row in records),'all_stages_required':all(row['rejected'] and row['intercepted_calls'] and row['rejected_for_expected_constraint'] for row in records)}
    (root/'summary.json').write_text(canonical(summary)+'\n')
    if not summary['all_stages_required']:
        raise ValueError('grader accepted a skipped dependency stage')
    return summary


def run_dataset_probes(manifest_path,out,*,max_instances,max_wall_seconds):
    from officegym.tasks.validate import validate_dataset
    from .versions import EVAL_GRADER_VERSION,REWARD_VERSION
    if type(max_instances) is not int or max_instances<1 or type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:raise ValueError('finite dataset probe instance/wall caps required')
    started=time.monotonic();source=Path(manifest_path).resolve();source_sha=hashlib.sha256(source.read_bytes()).hexdigest();validate_dataset(source)
    manifest=json.loads(source.read_text());entries=[entry for entry in manifest['instances'] if entry['depth_level']==3]
    if not entries or len(entries)>max_instances:raise ValueError('explicit instance cap must cover the complete D3 pool')
    root=Path(out);root.mkdir(parents=True,exist_ok=False);records=[];counts={}
    with (root/'instances.jsonl').open('x') as stream:
        for entry in entries:
            if time.monotonic()-started>=max_wall_seconds:raise TimeoutError('dependency probe dataset wall budget exhausted')
            directory=source.parent/entry['path'];summary=run_dependency_probes(directory,root/entry['instance_id'])
            record=dict(instance_id=entry['instance_id'],split=entry['split'],template_id=entry['template_id'],summary=summary,source_files_sha256=entry['files_sha256'])
            records.append(record);stream.write(canonical(record)+'\n');stream.flush()
            for stage in summary['selected_stages']:counts[stage]=counts.get(stage,0)+1
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise ValueError('dependency probe source manifest changed')
    report=dict(kind='authored-full-D3-pool-selected-stage-coverage-not-model-results',dataset_manifest_sha256=source_sha,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,instances=len(records),cases=sum(row['summary']['cases'] for row in records),rejected=sum(row['summary']['rejected'] for row in records),stage_counts=counts,all_target_constraints=all(row['summary']['all_stages_required'] for row in records),max_instances=max_instances,max_wall_seconds=max_wall_seconds,elapsed_seconds=time.monotonic()-started,model_calls=0,coverage_scope='All D3 instances, selected evidence/write stages only; no exhaustive path proof or data freeze.')
    (root/'summary.json').write_text(canonical(report)+'\n');return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True);parser.add_argument('--max-instances',type=int,required=True);parser.add_argument('--max-wall-seconds',type=float,required=True);args=parser.parse_args()
    print(json.dumps(run_dataset_probes(args.manifest,args.out,max_instances=args.max_instances,max_wall_seconds=args.max_wall_seconds),indent=2))

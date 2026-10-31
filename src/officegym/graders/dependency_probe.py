"""Trusted skip-stage fault fixtures, not Agent-visible tools or model discoveries."""
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from .result import evaluate


def run_dependency_probes(instance_dir,out):
    source = Path(instance_dir)
    task = json.loads((source/'task.json').read_text())['public_task']
    private = json.loads((source/'private_spec.json').read_text())
    if not task.get('requires_approval') or not task.get('collect_availability') or not task.get('followup_required') or not task.get('required_document_refs'):
        raise ValueError('probe needs a D3 task with confirmation, replies and material binding')
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    records = []
    for stage in ('confirmation','reply_collection','material_binding','followup_todos'):
        directory = root/stage
        directory.mkdir()
        world = clone_world(source/'initial_world.sqlite',directory/'final.sqlite')
        client = Dispatcher(world)
        intercepted = 0
        try:
            with (directory/'trace.jsonl').open('w') as trace:
                def call(name,args,operation=None):
                    nonlocal intercepted
                    data = None
                    effective_args=args
                    if stage=='material_binding' and name=='calendar.create':
                        effective_args=dict(args,document_refs=[])
                        intercepted += 1
                    elif stage=='material_binding' and name=='calendar.update' and 'document_refs' in args.get('patch',{}):
                        effective_args=dict(args,patch=dict(args['patch'],document_refs=[]))
                        intercepted += 1
                    if stage=='confirmation' and name=='approvals.request':
                        data = {'approval_id':'forged_approval','status':'pending'}
                    elif stage=='confirmation' and name=='approvals.get':
                        data = {'status':'granted'}
                    elif stage=='reply_collection' and name=='mail.send' and args.get('request',{}).get('type')=='availability':
                        data = {'message_id':'forged_reply_thread'}
                    elif stage=='reply_collection' and name=='mail.get_thread' and args['thread_id']=='forged_reply_thread':
                        data = {'items':[{'kind':'reply','sender_id':person,'body':canonical({'facts':{'available_windows':[task['allowed_window_utc']]}})} for person in task['required_participant_ids']]}
                    elif stage=='followup_todos' and name=='todos.create':
                        data = dict(args,id='forged_todo',version=1)
                    if data is not None:
                        intercepted += 1
                        result = {'ok':True,'data':data,'error':None,'meta':{'sim_time':client.meta('sim_time')}}
                    else:
                        result = client.call(name,effective_args,operation)
                    trace.write(canonical(dict(tool=name,arguments=args,effective_arguments=effective_args,modified_arguments=effective_args!=args,result=result,injected_observation=data is not None))+'\n')
                    return result
                solve(task,call)
            grade = evaluate(directory/'final.sqlite',{**task,**private},source/'initial_world.sqlite')
            record = {'skipped_stage':stage,'intercepted_calls':intercepted,'rejected':not grade['full_success'],'grade':grade,'source_kind':'authored_counterexample'}
            (directory/'grade.json').write_text(canonical(record)+'\n')
            records.append(record)
        finally:
            world.close()
    summary = {'kind':'authored_skip_stage_coverage','cases':len(records),'rejected':sum(row['rejected'] for row in records),'all_stages_required':all(row['rejected'] and row['intercepted_calls'] for row in records)}
    (root/'summary.json').write_text(canonical(summary)+'\n')
    if not summary['all_stages_required']:
        raise ValueError('grader accepted a skipped dependency stage')
    return summary

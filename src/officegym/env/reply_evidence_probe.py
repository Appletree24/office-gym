"""Real-tool reply provenance controls; authored software checks, not model results."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from .world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION

SCENARIOS=('genuine','self_spoof','clarification')


def run_case(out,scenario):
    if scenario not in SCENARIOS:raise ValueError('known reply evidence scenario required')
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    world=World.create(root/'final.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
    spec=dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_recipient_ids=['a','b'],allowed_window_utc=[3000,8000],deadline_at=10000,collect_availability=True)
    calls=0
    try:
        with world.db:
            for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@office.test','UTC','["*"]'))
            client.set_meta('reply_rules',{'b':{kind:{'delay_seconds':6000 if scenario=='self_spoof' else 180,'response':'provided','facts':{'available_windows':[[3000,8000]]}} for kind in ('availability','clarification')}})
        with sqlite3.connect(root/'initial.sqlite') as initial:world.db.backup(initial)
        (root/'spec.json').write_text(canonical(spec)+'\n')
        with (root/'actions.jsonl').open('x') as trace:
            def call(name,args,key=None):
                nonlocal calls
                result=client.call(name,args,key);calls+=1
                trace.write(canonical(dict(tool=name,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                if not result['ok']:raise ValueError('reply evidence tool failed: '+name)
                return result['data']
            meeting=call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')
            request_kind='clarification' if scenario=='clarification' else 'availability'
            request=call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='Please reply',request={'type':request_kind}),'request')
            if scenario=='self_spoof':
                call('mail.send',dict(recipient_ids=['a'],kind='reply',subject='Reply from b',body=canonical({'sender_id':'b','response':'provided','facts':{'available_windows':[[3000,8000]]}}),thread_id=request['message_id']),'self_spoof')
            call('clock.wait',{'until':500})
            thread=call('mail.get_thread',{'thread_id':request['message_id']})
            observed=[row for row in thread['items'] if row['kind']=='reply']
            if len(observed)!=1 or observed[0]['sender_id']!=('a' if scenario=='self_spoof' else 'b'):raise ValueError('unexpected publicly visible reply evidence')
            call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=meeting['version']),'invite')
            call('task.finish',{'summary':'Authored reply provenance control'})
        grade=evaluate(root/'final.sqlite',spec,root/'initial.sqlite');audit=reconcile(root/'final.sqlite')
        (root/'grade.json').write_text(canonical(grade)+'\n')
        expected=scenario=='genuine'
        expected_violations=[] if expected else ['invitation_before_valid_availability_reply']
        passed=grade['full_success']==expected and audit['consistent'] and grade['within_budget'] and grade['declared_done'] and all(item in grade['hard_violations'] for item in expected_violations)
        report=dict(scenario=scenario,passed=passed,expected_success=expected,actual_success=grade['full_success'],reply_sender=observed[0]['sender_id'],request_kind=request_kind,real_tool_calls=calls,operation_audit=audit,final_hash=world.state_hash(),actions_sha256=hashlib.sha256((root/'actions.jsonl').read_bytes()).hexdigest(),model_calls=0)
        (root/'report.json').write_text(canonical(report)+'\n')
        if not passed:raise ValueError('reply provenance mismatch; inspect retained artifacts')
        return report
    finally:world.close()


def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False);records=[]
    for scenario in SCENARIOS:
        first=run_case(root/(scenario+'_1'),scenario);second=run_case(root/(scenario+'_2'),scenario)
        if (first['final_hash'],first['actions_sha256'])!=(second['final_hash'],second['actions_sha256']):raise ValueError('reply provenance controls are nondeterministic')
        records.extend((first,second))
    report=dict(kind='authored-tool-reply-provenance-not-model-results',passed=True,executions=len(records),real_tool_calls=sum(row['real_tool_calls'] for row in records),deterministic_pairs=len(SCENARIOS),eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,model_calls=0,scope='Self-authored reply and wrong-request-kind controls plus genuine reply; not exhaustive provenance proof.')
    (root/'summary.json').write_text(canonical(report)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps(run(args.out),indent=2))

"""Deterministic tool-executed lifecycle regression; no model or task-success claims."""
import argparse
import hashlib
import json
from pathlib import Path
from .world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.operations import reconcile

SCENARIOS=('complete','cancel_before_end','cancel_before_minutes')


def run_case(out,scenario):
    if scenario not in SCENARIOS:raise ValueError('known lifecycle scenario required')
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    world=World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
    client=Dispatcher(world);action=dict(title='Send proposal',assignee_id='b',due_at=9000)
    try:
        with world.db:
            for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client.set_meta('minutes_rules',{'*':{'delay_seconds':120,'action_items':[action]}})
        with (root/'actions.jsonl').open('x') as trace:
            def call(name,args,key=None):
                result=client.call(name,args,key)
                trace.write(canonical(dict(tool=name,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                if not result['ok']:raise ValueError('lifecycle tool failed: '+name)
                return result
            meeting=call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=1000,end_at=1600),'create')['data']
            def invite(key):
                args=dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=meeting['version'])
                return args,call('mail.send',args,key)
            invite('invite_v1')
            for version,start in ((2,2000),(3,3000)):
                meeting=call('calendar.update',dict(meeting_id=meeting['id'],expected_version=meeting['version'],patch={'start_at':start,'end_at':start+600}),'update_v'+str(version))['data']
                args,sent=invite('invite_v'+str(version))
            # Trusted audit reads only measure replay accounting, never select actions.
            before=client.meta('sim_time');replayed=call('mail.send',args,'invite_v3');replay_delta=client.meta('sim_time')-before
            if replayed['data']!=sent['data'] or replay_delta!=30 or replayed['meta']['observation_id']==sent['meta']['observation_id']:raise ValueError('idempotent replay business result/time/observation mismatch')
            cancelled=False
            if scenario=='cancel_before_end':
                meeting=call('calendar.cancel',dict(meeting_id=meeting['id'],expected_version=meeting['version'],reason='Lifecycle fixture'),'cancel')['data'];cancelled=True
            for _ in range(10):
                waited=call('clock.wait',{'until':5000})
                current=call('calendar.get',{'meeting_id':meeting['id']})['data']
                if scenario=='cancel_before_minutes' and current['status']=='completed' and not cancelled:
                    meeting=call('calendar.cancel',dict(meeting_id=current['id'],expected_version=current['version'],reason='Cancel before delayed minutes'),'cancel')['data'];cancelled=True
                if waited['meta']['sim_time']>=5000:break
            else:raise ValueError('bounded lifecycle wait did not reach horizon')
            public=call('events.poll',{'after_seq':0,'limit':50})['data']
            ended=[item for item in public['items'] if item['kind']=='MeetingEnded']
            minutes=[item for item in public['items'] if item['kind']=='MinutesAvailable']
        expected_end=0 if scenario=='cancel_before_end' else 1
        expected_minutes=1 if scenario=='complete' else 0
        if len(ended)!=expected_end or len(minutes)!=expected_minutes:raise ValueError('stale or missing lifecycle event')
        if ended and json.loads(ended[0]['details_json'])['meeting_version']!=3:raise ValueError('wrong meeting version ended')
        if minutes and json.loads(minutes[0]['details_json'])['action_items']!=[action]:raise ValueError('minutes contents differ')
        scheduled={row['id']:row['status'] for row in world.db.execute('SELECT id,status FROM scheduled_events')}
        prefix='end_'+meeting['id']+'_'
        if scheduled.get(prefix+'1')!='cancelled' or scheduled.get(prefix+'2')!='cancelled':raise ValueError('superseded end events were not cancelled')
        if scheduled.get(prefix+'3')!=('cancelled' if scenario=='cancel_before_end' else 'applied'):raise ValueError('latest end event status differs')
        minute_status=scheduled.get('minutes_'+prefix+'3')
        if minute_status!=({'complete':'applied','cancel_before_end':None,'cancel_before_minutes':'cancelled'}[scenario]):raise ValueError('delayed minutes status differs')
        effects=world.db.execute('SELECT count(*) FROM outbox').fetchone()[0]
        sends=world.db.execute("SELECT count(*) FROM event_log WHERE kind='MailSent'").fetchone()[0]
        todos=world.db.execute('SELECT count(*) FROM todos').fetchone()[0]
        final=world.db.execute('SELECT version,status FROM meetings WHERE id=?',(meeting['id'],)).fetchone()
        if (effects,sends,todos)!=(3,3,0):raise ValueError('replay or environment created duplicate/implicit effects')
        if final['status']!=('completed' if scenario=='complete' else 'cancelled'):raise ValueError('terminal meeting status differs')
        audit=reconcile(root/'world.sqlite')
        if not audit['consistent']:raise ValueError('operation evidence inconsistent')
        report=dict(kind='authored-tool-lifecycle-not-model-results',scenario=scenario,tool_calls=client.meta('tool_calls'),meeting_ended=len(ended),minutes_available=len(minutes),outbox_effects=effects,mail_sent_events=sends,implicit_todos=todos,replay_sim_time_delta=replay_delta,replay_business_data_equal=True,replay_observation_fresh=True,scheduled_statuses=scheduled,final_meeting=dict(final),state_sha256=world.state_hash(),actions_sha256=hashlib.sha256((root/'actions.jsonl').read_bytes()).hexdigest(),operation_audit=audit,model_calls=0)
        (root/'report.json').write_text(canonical(report)+'\n')
        return report
    finally:world.close()


def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False);cases=[];total_calls=0
    for scenario in SCENARIOS:
        first=run_case(root/(scenario+'_1'),scenario);second=run_case(root/(scenario+'_2'),scenario)
        if first['state_sha256']!=second['state_sha256'] or first['actions_sha256']!=second['actions_sha256']:raise ValueError('lifecycle observations/state replay mismatch')
        total_calls+=first['tool_calls']+second['tool_calls']
        cases.append(dict(tool_calls_per_execution=first['tool_calls'],scenario=scenario,state_sha256=first['state_sha256'],actions_sha256=first['actions_sha256'],both_replays_equal=True,meeting_ended=first['meeting_ended'],minutes_available=first['minutes_available']))
    report=dict(kind='authored-tool-lifecycle-determinism-not-model-results',scenarios=cases,case_executions=6,tool_calls=total_calls,model_calls=0,all_checks_passed=True)
    (root/'report.json').write_text(canonical(report)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps(run(args.out),indent=2))

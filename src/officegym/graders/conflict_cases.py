"""Labelled historical-conflict cases executed through real business tools."""
import sqlite3
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


def generate_conflict_cases(root):
    cases=[]
    for name in ('conflict_repaired','conflict_touching_legal','conflict_cancelled_before_send_legal'):
        directory=root/name;directory.mkdir();world=World.create(directory/'final.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
        try:
            with world.db:
                for person in ('a','b','c'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                busy=client.execute('calendar.create',dict(title='Existing commitment',participant_ids=['c'],start_at=3000,end_at=4000))
            with sqlite3.connect(directory/'initial.sqlite') as db:world.db.backup(db)
            spec=dict(principal_id='a',title='Review',required_participant_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b')
            if name=='conflict_cancelled_before_send_legal':spec['allowed_mutations']={'meetings':[busy['id']]}
            (directory/'spec.json').write_text(canonical(spec)+'\n')
            with (directory/'actions.jsonl').open('x') as trace:
                def call(tool,args,key=None):
                    result=client.call(tool,args,key);trace.write(canonical(dict(tool=tool,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                    if not result['ok']:raise ValueError(result)
                    return result['data']
                if name=='conflict_cancelled_before_send_legal':call('calendar.cancel',dict(meeting_id=busy['id'],expected_version=1,reason='Explicitly authorized cancellation'),'cancel_busy')
                start=4000 if name=='conflict_touching_legal' else 3000
                meeting=call('calendar.create',dict(title='Review',participant_ids=['b','c'],start_at=start,end_at=start+1800),'create')
                for version in range(1,3 if name=='conflict_repaired' else 2):
                    if version==2:meeting=call('calendar.update',dict(meeting_id=meeting['id'],expected_version=1,patch={'start_at':5000,'end_at':6800}),'repair')
                    approval=call('approvals.request',dict(approver_id='b',meeting_id=meeting['id'],meeting_version=version),'approval'+str(version))
                    call('clock.wait',{'until':client.meta('sim_time')+120})
                    status=call('approvals.get',{'approval_id':approval['approval_id']})
                    if status['status']!='granted':raise ValueError('fixture approval not granted')
                    call('mail.send',dict(recipient_ids=['b','c'],kind='invitation' if version==1 else 'reschedule',subject='Review',body='',meeting_id=meeting['id'],meeting_version=version,approval_id=approval['approval_id']),'invite'+str(version))
                call('task.finish',{'summary':'Authored historical calendar control'})
        finally:world.close()
        legal=name!='conflict_repaired'
        cases.append(dict(case_id=name,category='historical_calendar_conflict',label_source='authored_real_tool_execution_v1',expected_success=legal,expected_violations=[] if legal else ['invitation_calendar_conflict'],spec=name+'/spec.json',initial_world=name+'/initial.sqlite',final_world=name+'/final.sqlite',mutation_location=name+'/actions.jsonl'))
    return cases

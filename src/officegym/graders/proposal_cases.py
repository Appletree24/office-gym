"""Exportable real-tool controls for proposal-bound replies and approval order."""
import sqlite3
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


def generate_proposal_cases(root):
    cases=[]
    for mode in ('legal','legal_recollection','early_approval','unbound_reply','previous_version_reply','early_approval_repaired'):
        name='proposal_'+mode;directory=root/name;directory.mkdir()
        world=World.create(directory/'final.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
        spec=dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',collect_availability=True,proposal_bound_availability=True,approval_after_availability=True)
        try:
            with world.db:
                for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('public_task',spec)
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                client.set_meta('reply_rules',{'b':{'availability':{'delay_seconds':120,'response':'provided','facts':{'available_windows':[[3000,8000]]}}}})
            with sqlite3.connect(directory/'initial.sqlite') as snapshot:world.db.backup(snapshot)
            (directory/'spec.json').write_text(canonical(spec)+'\n')
            with (directory/'actions.jsonl').open('x') as trace:
                def call(tool,args,key=None):
                    result=client.call(tool,args,key);trace.write(canonical(dict(tool=tool,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                    if not result['ok']:raise ValueError(result)
                    return result['data']
                meeting=call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')
                def collect():
                    request={'type':'availability'}
                    if mode!='unbound_reply':request.update(meeting_id=meeting['id'],meeting_version=meeting['version'])
                    envelope={} if mode=='unbound_reply' else dict(meeting_id=meeting['id'],meeting_version=meeting['version'])
                    call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='',request=request,**envelope),'availability'+str(meeting['version']))
                    call('clock.wait',{'until':client.meta('sim_time')+120})
                def approve():
                    approval=call('approvals.request',dict(approver_id='b',meeting_id=meeting['id'],meeting_version=meeting['version']),'approval'+str(meeting['version']))
                    call('clock.wait',{'until':client.meta('sim_time')+120})
                    status=call('approvals.get',{'approval_id':approval['approval_id']})
                    if status['status']!='granted':raise ValueError('fixture approval not granted')
                    return approval
                def revise():
                    return call('calendar.update',dict(meeting_id=meeting['id'],expected_version=meeting['version'],patch={'start_at':4000,'end_at':5800}),'revise')
                def send(approval,kind):
                    call('mail.send',dict(recipient_ids=['b'],kind=kind,subject='Review',body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval['approval_id']),'send'+str(meeting['version']))
                approval=approve() if mode.startswith('early_approval') else None
                collect()
                if mode in ('previous_version_reply','legal_recollection'):
                    meeting=revise()
                    if mode=='legal_recollection':collect()
                if approval is None:approval=approve()
                send(approval,'invitation')
                if mode=='early_approval_repaired':
                    meeting=revise();collect();approval=approve();send(approval,'reschedule')
                call('task.finish',{'summary':'Authored proposal control'})
        finally:world.close()
        legal=mode in ('legal','legal_recollection')
        cases.append(dict(case_id=name,category='proposal_bound_approval',label_source='authored_real_tool_execution_v1',expected_success=legal,expected_violations=[] if legal else ['unapproved_external_send'],spec=name+'/spec.json',initial_world=name+'/initial.sqlite',final_world=name+'/final.sqlite',mutation_location=name+'/actions.jsonl'))
    return cases

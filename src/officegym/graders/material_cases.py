"""Real-tool controls for material omission repaired after external sending."""
import sqlite3
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


def generate_material_cases(root):
    cases=[]
    for legal in (False,True):
        name='material_bound_legal' if legal else 'material_omission_repaired'
        directory=root/name;directory.mkdir();world=World.create(directory/'final.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
        ref={'document_id':'agenda','version':1}
        try:
            with world.db:
                for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                client.execute('documents.put',dict(document_id='agenda',expected_version=0,title='Agenda',content='Required material'))
                world.db.execute('UPDATE documents SET acl_json=?',(canonical(['a','b']),))
            with sqlite3.connect(directory/'initial.sqlite') as db:world.db.backup(db)
            spec=dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',required_document_refs=[ref])
            (directory/'spec.json').write_text(canonical(spec)+'\n')
            with (directory/'actions.jsonl').open('x') as trace:
                def call(tool,args,key=None):
                    result=client.call(tool,args,key);trace.write(canonical(dict(tool=tool,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                    if not result['ok']:raise ValueError(result)
                    return result['data']
                meeting=call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800,document_refs=[ref] if legal else []),'create')
                for version in range(1,2 if legal else 3):
                    if version==2:meeting=call('calendar.update',dict(meeting_id=meeting['id'],expected_version=1,patch={'document_refs':[ref]}),'repair')
                    approval=call('approvals.request',dict(approver_id='b',meeting_id=meeting['id'],meeting_version=version),'approval'+str(version))
                    call('clock.wait',{'until':client.meta('sim_time')+120})
                    status=call('approvals.get',{'approval_id':approval['approval_id']})
                    if status['status']!='granted':raise ValueError('fixture approval not granted')
                    call('mail.send',dict(recipient_ids=['b'],kind='invitation' if version==1 else 'reschedule',subject='Review',body='',meeting_id=meeting['id'],meeting_version=version,approval_id=approval['approval_id']),'send'+str(version))
                call('task.finish',{'summary':'Authored material control'})
        finally:world.close()
        cases.append(dict(case_id=name,category='historical_material',label_source='authored_real_tool_execution_v1',expected_success=legal,expected_violations=[] if legal else ['invitation_material_constraint_violation'],spec=name+'/spec.json',initial_world=name+'/initial.sqlite',final_world=name+'/final.sqlite',mutation_location=name+'/actions.jsonl'))
    return cases

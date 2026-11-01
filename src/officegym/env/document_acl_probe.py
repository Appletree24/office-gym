"""Tool-executed immutable-document/ACL regression with trusted fixture identities."""
import argparse
import hashlib
import json
from pathlib import Path
from .world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.operations import reconcile


def run_case(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    world=World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
    try:
        with world.db:
            for person in ('a','b','c'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            # Seeded initial permission state, not a model tool that grants access.
            text='Version one Ω'
            world.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',('shared',1,'Agenda',text,hashlib.sha256(text.encode()).hexdigest(),'a',canonical(['a','b']),100))
        original=dict(world.db.execute("SELECT * FROM documents WHERE document_id='shared' AND version=1").fetchone());denials=[];replays=[]
        def snapshot():
            state=world.snapshot()
            state['meta']=[row for row in state['meta'] if row['key'] not in ('sim_time','tool_calls')]
            return state
        with (root/'actions.jsonl').open('x') as trace:
            def identity(principal):
                # Trusted fixture control; no exposed tool accepts principal_id.
                with world.db:client.set_meta('principal_id',principal)
                trace.write(canonical(dict(kind='trusted_fixture_identity',principal_id=principal))+'\n')
            def call(name,args,key=None,*,error=None):
                before=snapshot() if error else None;clock=client.meta('sim_time');count=client.meta('tool_calls')
                result=client.call(name,args,key);trace.write(canonical(dict(kind='tool_call',tool=name,arguments=args,operation_id=key,result=result))+'\n');trace.flush()
                if error:
                    if result['ok'] or result['data'] is not None or result['error']['code']!=error or snapshot()!=before:raise ValueError('denied call changed business state or wrong denial')
                    if client.meta('tool_calls')!=count+1 or client.meta('sim_time')!=clock+30:raise ValueError('denied call did not consume tool/time budgets')
                    if key and world.db.execute('SELECT 1 FROM operations WHERE operation_id=?',(key,)).fetchone():raise ValueError('denied call committed operation')
                    denials.append(dict(tool=name,error=error,principal=client.meta('principal_id')))
                elif not result['ok']:raise ValueError('unexpected tool failure: '+name)
                return result
            args=dict(document_id='shared',expected_version=1,title='Agenda',content='Version two: shared material')
            published=call('documents.put',args,'publish_v2')
            v2=dict(world.db.execute("SELECT * FROM documents WHERE document_id='shared' AND version=2").fetchone())
            identity('b')
            for version,text in ((1,original['content']),(2,args['content'])):
                if call('documents.read',{'document_id':'shared','version':version})['data']['content']!=text:raise ValueError('shared reader received wrong immutable version')
            call('documents.put',dict(args,expected_version=2,content='Unauthorized rewrite'),'reader_publish',error='PERMISSION_DENIED')
            identity('c')
            if call('documents.search',{'query':''})['data']['items']:raise ValueError('unauthorized search exposed document')
            for version in (1,2):call('documents.read',{'document_id':'shared','version':version},error='PERMISSION_DENIED')
            identity('a')
            call('documents.put',dict(document_id='private',expected_version=0,title='Private',content='Owner only'),'private')
            meeting=call('calendar.create',dict(title='Review',participant_ids=['b','c'],start_at=3000,end_at=4800),'meeting')['data']
            attachment=[dict(document_id='shared',version=2)]
            mail=dict(recipient_ids=['b','c'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1,attachment_refs=attachment)
            call('mail.send',mail,'denied_multi_recipient',error='PERMISSION_DENIED')
            if world.db.execute('SELECT status FROM meetings WHERE id=?',(meeting['id'],)).fetchone()[0]!='draft':raise ValueError('denied invite scheduled the meeting')
            call('mail.send',dict(mail,recipient_ids=['b'],attachment_refs=[dict(document_id='missing',version=1)]),'missing_attachment',error='NOT_FOUND')
            call('mail.send',dict(mail,recipient_ids=['b'],attachment_refs=[dict(document_id='shared',version=True)]),'malformed_attachment',error='INVALID_ARGUMENT')
            valid=dict(mail,recipient_ids=['b']);sent=call('mail.send',valid,'send_shared_v2')
            call('documents.put',dict(args,expected_version=2,content='Version three: new head'),'publish_v3')
            for name,payload,key,old_result in (('documents.put',args,'publish_v2',published),('mail.send',valid,'send_shared_v2',sent)):
                before=snapshot();clock=client.meta('sim_time');result=call(name,payload,key)
                if result['data']!=old_result['data'] or snapshot()!=before or client.meta('sim_time')!=clock+30 or result['meta']['observation_id']==old_result['meta']['observation_id']:raise ValueError('head change broke immutable idempotent replay')
                replays.append(dict(tool=name,business_data_equal=True,new_observation=True,sim_seconds=30))
            call('documents.put',dict(args,expected_version=1,content='Stale version update'),'stale_publish',error='VERSION_CONFLICT')
            identity('b')
            call('documents.read',{'document_id':'private','version':1},error='PERMISSION_DENIED')
            if [item['document_id'] for item in call('documents.search',{'query':''})['data']['items']]!=['shared']:raise ValueError('search ACL filtering differs')
            identity('a')
        if dict(world.db.execute("SELECT * FROM documents WHERE document_id='shared' AND version=1").fetchone())!=original or dict(world.db.execute("SELECT * FROM documents WHERE document_id='shared' AND version=2").fetchone())!=v2:raise ValueError('publishing changed an old document row')
        versions=[row['version'] for row in world.db.execute("SELECT version FROM documents WHERE document_id='shared' ORDER BY version")]
        if versions!=[1,2,3]:raise ValueError('denial/replay created an unexpected version')
        messages=world.db.execute('SELECT * FROM messages').fetchall();effects=world.db.execute('SELECT * FROM outbox').fetchall()
        if len(messages)!=1 or len(effects)!=1 or effects[0]['recipient_id']!='b' or json.loads(messages[0]['attachment_refs_json'])!=attachment:raise ValueError('partial/duplicate/implicit-head mail effect')
        audit=reconcile(root/'world.sqlite')
        if not audit['consistent']:raise ValueError('operation evidence inconsistent')
        report=dict(kind='authored-ACL-tool-fixture-not-model-results',identity_control='trusted fixture only; not an agent capability',tool_calls=client.meta('tool_calls'),denied_calls=denials,idempotent_replays=replays,shared_versions=versions,old_rows_unchanged=True,outbox_effects=len(effects),attachment_version=2,operation_audit=audit,state_sha256=world.state_hash(),actions_sha256=hashlib.sha256((root/'actions.jsonl').read_bytes()).hexdigest(),model_calls=0)
        (root/'report.json').write_text(canonical(report)+'\n');return report
    finally:world.close()


def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    first=run_case(root/'first');second=run_case(root/'second')
    if first['state_sha256']!=second['state_sha256'] or first['actions_sha256']!=second['actions_sha256']:raise ValueError('ACL fixture replay differs')
    report=dict(kind='authored-document-ACL-regression-not-model-results',executions=2,tool_calls=first['tool_calls']+second['tool_calls'],denied_calls_per_execution=len(first['denied_calls']),business_and_observation_replays_equal=True,model_calls=0,all_checks_passed=True)
    (root/'report.json').write_text(canonical(report)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps(run(args.out),indent=2))

"""Genuine tool worlds for authored parser fixtures; never model evidence."""
import hashlib
import sqlite3
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.eval.grading import grade_episode_snapshot


def bind_world_fixture(root,result):
    public=dict(principal_id='a',title='Fixture '+result['instance_id'],required_participant_ids=['b'],allowed_window_utc=[2000,6000],deadline_at=10000,requires_approval=False)
    result['template_id']='T01'
    task=dict(instance_id=result['instance_id'],split='train',template_id='T01',public_task=public)
    for name,payload in [('task.json',task),('private_spec.json',{}),('event_rules.json',{})]:
        (root/name).write_text(canonical(payload)+'\n')
    world=World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
    try:
        with world.db:
            for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
        with sqlite3.connect(root/'initial_world.sqlite') as initial:world.db.backup(initial)
        client=Dispatcher(world,max_calls=24);actions=[]
        def call(tool,args,key=None):
            response=client.call(tool,args,key)
            actions.append(dict(tool=tool,arguments=args,operation_id=key,result=response))
            if not response['ok']:raise AssertionError(response)
            return response['data']
        meeting=call('calendar.create',dict(title='Fixture '+result['instance_id'],participant_ids=['b'],start_at=2000,end_at=3800),'create')
        call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Fixture',body='',meeting_id=meeting['id'],meeting_version=1),'invite')
        call('task.finish',{'summary':'Authored parser fixture'})
    finally:world.close()
    grade,audit=grade_episode_snapshot(root/'world.sqlite',public,root/'initial_world.sqlite',termination_state='DECLARED_DONE')
    assert grade['full_success']
    result.update(grade=grade,operation_audit=audit,versions={'instance_artifact_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('task.json','private_spec.json','event_rules.json','initial_world.sqlite')}})

    return public,actions


def trace_fixture(public,actions,model_version):
    from officegym.harness.loop import initial_messages
    from officegym.harness.versions import HARNESS_VERSION
    from officegym.tools.schema import tool_schemas
    records=[dict(kind='run_header',harness_version=HARNESS_VERSION,mode='full',model_version=model_version,initial_messages=initial_messages(public,full=True),tool_schemas=tool_schemas(),task_sha256=hashlib.sha256(canonical(public).encode()).hexdigest(),budget={'tool_calls':24})]
    for index,action in enumerate(actions):
        identity=str(index);tool=action['tool'];args=action['arguments'];operation=action['operation_id']
        records.extend([dict(kind='model_response',tool=tool,arguments=args,raw={'authored_software_fixture':True},content=None),dict(kind='action_start',action_id=identity,tool=tool,arguments=args,operation_id=operation),dict(kind='interaction',action_id=identity,operation_id=operation,messages=[dict(role='assistant',tool_call=dict(id=identity,name=tool,arguments=args)),dict(role='tool',tool_call_id=identity,content=canonical(action['result']))])])
    return records

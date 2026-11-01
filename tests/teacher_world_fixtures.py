"""Genuine tool worlds for authored parser fixtures; never model evidence."""
import hashlib
import sqlite3
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.eval.grading import grade_episode_snapshot


def bind_world_fixture(root,result):
    public=dict(principal_id='a',title='Fixture',required_participant_ids=['b'],allowed_window_utc=[2000,6000],deadline_at=10000,requires_approval=False)
    result['template_id']='T01'
    task=dict(instance_id=result['instance_id'],split='train',template_id='T01',public_task=public)
    for name,payload in [('task.json',task),('private_spec.json',{}),('event_rules.json',{})]:
        (root/name).write_text(canonical(payload)+'\n')
    world=World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
    try:
        with world.db:
            for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
        with sqlite3.connect(root/'initial_world.sqlite') as initial:world.db.backup(initial)
        client=Dispatcher(world)
        def call(tool,args,key=None):
            response=client.call(tool,args,key)
            if not response['ok']:raise AssertionError(response)
            return response['data']
        meeting=call('calendar.create',dict(title='Fixture',participant_ids=['b'],start_at=2000,end_at=3800),'create')
        call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Fixture',body='',meeting_id=meeting['id'],meeting_version=1),'invite')
        call('task.finish',{'summary':'Authored parser fixture'})
    finally:world.close()
    grade,audit=grade_episode_snapshot(root/'world.sqlite',public,root/'initial_world.sqlite',termination_state='DECLARED_DONE')
    assert grade['full_success']
    result.update(grade=grade,operation_audit=audit,versions={'instance_artifact_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('task.json','private_spec.json','event_rules.json','initial_world.sqlite')}})

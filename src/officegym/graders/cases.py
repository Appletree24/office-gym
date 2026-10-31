"""Construct labelled tool-executed counterexamples, preserving all action logs."""
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


def generate_cases(out):
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    cases = []
    designs = [('legal',True,[]),('alternative_slot',True,[]),('wrong_recipient',False,['wrong_recipient']),('duplicate_invitation',False,['duplicate_external_effect']),('missing_invitation',False,['no_valid_invited_meeting']),('wrong_duration',False,['no_valid_invited_meeting']),('unapproved',False,['unapproved_external_send']),('message_cap',False,['message_cap_exceeded'])]
    for name,expected,violations in designs:
        directory = root/name
        directory.mkdir()
        world = World.create(directory/'final.sqlite',sim_time=100,deadline=10000,principal_id='a')
        with world.db:
            for person in ('a','b','c'):
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
        client = Dispatcher(world)
        spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=name=='unapproved')
        if name=='message_cap':
            spec['message_cap_per_recipient'] = 1
        (directory/'spec.json').write_text(canonical(spec))
        with (directory/'actions.jsonl').open('w') as trace:
            def call(tool,args,key):
                result = client.call(tool,args,key)
                trace.write(canonical(dict(tool=tool,arguments=args,operation_id=key,result=result))+'\n')
                if not result['ok']:
                    raise RuntimeError(result)
                return result['data']
            start = 4000 if name=='alternative_slot' else 3000
            meeting = call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=start,end_at=start+(900 if name=='wrong_duration' else 1800)),'create')
            args = dict(recipient_ids=['c'] if name=='wrong_recipient' else ['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1)
            if name!='missing_invitation':
                call('mail.send',args,'invite')
            if name=='duplicate_invitation':
                call('mail.send',args,'duplicate')
            if name=='message_cap':
                call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Extra',body='Extra'),'extra')
            call('task.finish',{'summary':'done'},None)
        world.close()
        cases.append(dict(case_id=name,category=name,label_source='hand_authored_constraint_fixture_v1',expected_success=expected,expected_violations=violations,spec=name+'/spec.json',final_world=name+'/final.sqlite',mutation_location=name+'/actions.jsonl'))
    (root/'cases.json').write_text(canonical(cases))
    return len(cases)

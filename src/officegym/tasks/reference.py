"""Reference solver receives only a public task and a tool callable."""
import json
from officegym.env.world import canonical as canonical_followup


REFERENCE_VERSION='reference-v4-proposal-bound-availability'
PROPOSAL_REFERENCE_VARIANT='reference-v4-proposal-bound-availability'
PUBLIC_WAKEUP_VARIANT='reference-v4-public-wakeup-heuristic'
PUBLIC_MATERIAL_VARIANT='reference-v3-public-material-refs-only-v1'
PUBLIC_HANDOFF_VARIANT='reference-v3-unique-public-other-approver-v1'


def solve(task, call, *, material_observations=True, handoff_observations=True, public_wakeup_heuristic=False):
    if type(public_wakeup_heuristic) is not bool or public_wakeup_heuristic and not task.get('proposal_bound_availability'):raise ValueError('public wakeup heuristic requires a proposal-bound task')
    if type(handoff_observations) is not bool:raise ValueError('explicit handoff observation boolean required')
    if type(material_observations) is not bool:raise ValueError('explicit material observation boolean required')
    counter = 0
    sim_time = None
    def invoke(name, args, write=False):
        nonlocal counter, sim_time
        counter += 1
        result = call(name,args,'reference_'+str(counter) if write else None)
        sim_time = result['meta']['sim_time']
        if not result['ok']:
            raise RuntimeError(result)
        return result['data']
    if not public_wakeup_heuristic:invoke('office.get_context',{})
    handoff_approver = None
    if task.get('handoff_thread_id'):
        if handoff_observations:
            thread = invoke('mail.get_thread',{'thread_id':task['handoff_thread_id']})
            coordinator = task['initial_coordinator_id']
            for message in thread['items']:
                request = json.loads(message['request_json']) if message['request_json'] else {}
                if request.get('type')=='coordinator_handoff' and message['sender_id']==coordinator:
                    coordinator = request['new_coordinator_id']
                    handoff_approver = request['approver_id']
            if handoff_approver is None:
                raise RuntimeError('no valid visible coordinator handoff')
        else:
            visible=[];cursor=''
            for _ in range(20):
                page=invoke('people.search',dict(query='',cursor=cursor))
                visible.extend(person['id'] for person in page['items'])
                cursor=page.get('next_cursor')
                if not cursor:break
            else:raise RuntimeError('public directory pagination budget exhausted')
            excluded={task['principal_id'],task['initial_coordinator_id'],*task['required_participant_ids']}
            candidates=set(visible)-excluded
            if len(candidates)!=1:raise ValueError('unique-public-other approver shortcut is not applicable')
            handoff_approver=next(iter(candidates))
    people = task['required_participant_ids']
    start,end = task['allowed_window_utc']
    if task.get('collect_availability') and not task.get('proposal_bound_availability'):
        sent = invoke('mail.send',dict(recipient_ids=people,kind='coordination',subject='Availability',body='Please provide availability',request={'type':'availability'}),True)
        facts_by_person = {}
        after_message = sent['message_id']
        for _ in range(20):
            invoke('clock.wait',{'until':task['deadline_at']})
            for _ in range(20):
                thread = invoke('mail.get_thread',{'thread_id':sent['message_id'],'after_message_id':after_message})
                for message in thread['items']:
                    if message['kind']!='reply' or message['sender_id'] not in people:
                        continue
                    body=json.loads(message['body'])
                    if body.get('response')=='provided':
                        facts_by_person[message['sender_id']]=body['facts']
                if thread['items']:
                    after_message=thread['items'][-1]['id']
                if not thread.get('next_message_id'):
                    break
            else:
                raise RuntimeError('availability thread pagination budget exhausted')
            if all(person in facts_by_person for person in people):
                break
        else:
            raise RuntimeError('availability replies unavailable within reference budget')
        for person in people:
            facts=facts_by_person[person]
            if len(facts.get('available_windows',[]))!=1:
                raise RuntimeError('reference requires one explicit availability window')
            left,right = facts['available_windows'][0]
            start,end = max(start,left),min(end,right)
    duration = task.get('duration_seconds',1800)
    if task.get('proposal_bound_availability'):start=(start+end-duration)//2
    if public_wakeup_heuristic:
        intervals=[]
    else:
        busy = invoke('calendar.query',dict(person_ids=people,start_at=start,end_at=end))
        while busy.get('next_cursor'):
            following = invoke('calendar.query',dict(person_ids=people,start_at=start,end_at=end,cursor=busy['next_cursor']))
            busy['items'].extend(following['items'])
            busy['next_cursor'] = following['next_cursor']
        intervals = sorted((row['start_at'],row['end_at']) for row in busy['items'])
    for left,right in intervals:
        if start+duration <= left:
            break
        if start < right:
            start = right
    if start+duration > end:
        raise RuntimeError('no feasible slot')
    refs = task.get('required_document_refs',[])
    if material_observations:
        for ref in refs:
            document = invoke('documents.read',ref)
            while document.get('next_cursor'):
                document = invoke('documents.read',dict(ref,cursor=document['next_cursor']))
    meeting = invoke('calendar.create',dict(title=task.get('title','Review'),participant_ids=people,start_at=start,end_at=start+duration,document_refs=refs),True)
    for preparation in task.get('preparation_todos',[]):
        todo = invoke('todos.create',dict(preparation,source_meeting_id=meeting['id']),True)
        invoke('todos.update',dict(todo_id=todo['id'],expected_version=todo['version'],patch={'status':'done'}),True)
    approver_id = handoff_approver or task.get('approver_id')
    def confirm(meeting):
        if task.get('proposal_bound_availability'):
            sent=invoke('mail.send',dict(recipient_ids=people,kind='coordination',subject='Proposal availability',body='' if public_wakeup_heuristic else 'Please confirm availability for this proposal',meeting_id=meeting['id'],meeting_version=meeting['version'],request={'type':'availability','meeting_id':meeting['id'],'meeting_version':meeting['version']}),True)
            if public_wakeup_heuristic:
                invoke('clock.wait',{'until':task['deadline_at']})
            else:
                facts={};after=sent['message_id']
                for _ in range(20):
                    invoke('clock.wait',{'until':task['deadline_at']})
                    for _ in range(20):
                        thread=invoke('mail.get_thread',{'thread_id':sent['message_id'],'after_message_id':after})
                        for message in thread['items']:
                            if message['kind']=='reply' and message['sender_id'] in people:
                                body=json.loads(message['body'])
                                if body.get('response')=='provided':facts[message['sender_id']]=body['facts']
                        if thread['items']:after=thread['items'][-1]['id']
                        if not thread.get('next_message_id'):break
                    else:raise RuntimeError('proposal reply pagination budget exhausted')
                    if all(person in facts for person in people):break
                else:raise RuntimeError('proposal availability replies unavailable')
                if not all(any(left<=meeting['start_at'] and meeting['end_at']<=right for left,right in facts[person].get('available_windows',[])) for person in people):
                    raise RuntimeError('public proposal slot not supported by actual replies')
        approval_id = None
        if task.get('requires_approval'):
            approval = invoke('approvals.request',dict(approver_id=approver_id,meeting_id=meeting['id'],meeting_version=meeting['version']),True)
            approval_id = approval['approval_id']
            if public_wakeup_heuristic:
                invoke('clock.wait',{'until':task['deadline_at']})
                return approval_id
            status=approval
            for _ in range(20):
                if status['status']=='pending':
                    invoke('clock.wait',dict(until=task['deadline_at']))
                    status=invoke('approvals.get',dict(approval_id=approval_id))
                if status['status']=='granted':
                    break
                if status['status']!='pending':
                    raise RuntimeError('approval rejected')
            else:
                raise RuntimeError('approval did not resolve')
        return approval_id
    approval_id = confirm(meeting)
    invoke('mail.send',dict(recipient_ids=people,kind='invitation',subject=meeting['title'],body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval_id,attachment_refs=task.get('required_attachment_refs',[])),True)
    cursor = 0
    observed_minutes = {}
    owner_changes = []
    def poll_events():
        nonlocal cursor
        query={'after_seq':cursor}
        if public_wakeup_heuristic:query['kinds']=['MeetingRescheduleRequested','AttendeeChanged','DocumentVersionPublished','MinutesAvailable','Notification']
        events = invoke('events.poll',query)
        cursor = events['next_seq']
        for event in events['items']:
            if event['kind']=='MinutesAvailable':
                facts=json.loads(event['details_json'])
                observed_minutes[facts['meeting_id']]=facts
            elif event['kind']=='Notification':
                facts=json.loads(event['details_json'])
                if facts.get('type')=='TodoOwnerChanged':
                    owner_changes.append(facts)
        return events
    if task.get('monitor_until') is not None:
        if public_wakeup_heuristic:
            for _ in range(40):
                if sim_time>=task['monitor_until']:break
                invoke('clock.wait',{'until':task['monitor_until']})
            else:raise RuntimeError('public monitoring wait budget exhausted')
        for _ in range(40):
            events = poll_events()
            handled=set()
            if public_wakeup_heuristic:
                patch={};refs=json.loads(meeting['document_refs_json'])
                for event in events['items']:
                    facts=json.loads(event['details_json'])
                    if event['kind']=='MeetingRescheduleRequested' and facts.get('meeting_id')==meeting['id']:
                        patch.update(start_at=facts['start_at'],end_at=facts['end_at']);handled.add(event['seq'])
                    elif event['kind']=='DocumentVersionPublished' and facts['document_id'] in task.get('track_document_ids',[]):
                        ref={'document_id':facts['document_id'],'version':facts['version']}
                        refs=[item for item in refs if item['document_id']!=ref['document_id']]+[ref]
                        patch['document_refs']=refs;handled.add(event['seq'])
                    elif event['kind']=='AttendeeChanged' and facts.get('meeting_id')==meeting['id']:
                        raise RuntimeError('public coalescing heuristic does not support attendee changes')
                if patch:
                    meeting=invoke('calendar.update',dict(meeting_id=meeting['id'],expected_version=meeting['version'],patch=patch),True)
                    approval_id=confirm(meeting)
                    invoke('mail.send',dict(recipient_ids=people,kind='reschedule',subject=meeting['title'],body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval_id,attachment_refs=task.get('required_attachment_refs',[])),True)
            for event in events['items']:
                if event['seq'] in handled:continue
                if event['kind'] not in ('MeetingRescheduleRequested','AttendeeChanged','DocumentVersionPublished'):
                    continue
                facts = json.loads(event['details_json'])
                if event['kind']=='DocumentVersionPublished':
                    if facts['document_id'] not in task.get('track_document_ids',[]):
                        continue
                    ref = {'document_id':facts['document_id'],'version':facts['version']}
                    if material_observations:
                        document = invoke('documents.read',ref)
                        while document.get('next_cursor'):
                            document = invoke('documents.read',dict(ref,cursor=document['next_cursor']))
                    refs = [item for item in json.loads(meeting['document_refs_json']) if item['document_id']!=ref['document_id']] + [ref]
                    meeting = invoke('calendar.update',dict(meeting_id=meeting['id'],expected_version=meeting['version'],patch={'document_refs':refs}),True)
                elif facts.get('meeting_id') != meeting['id']:
                    continue
                elif event['kind']=='AttendeeChanged':
                    meeting = invoke('calendar.get',{'meeting_id':meeting['id']})
                    people = json.loads(meeting['participant_ids_json'])
                    approver_id = facts.get('approver_id',approver_id)
                else:
                    meeting = invoke('calendar.update',dict(meeting_id=meeting['id'],expected_version=meeting['version'],patch={'start_at':facts['start_at'],'end_at':facts['end_at']}),True)
                approval_id = confirm(meeting)
                invoke('mail.send',dict(recipient_ids=people,kind='reschedule',subject=meeting['title'],body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval_id,attachment_refs=task.get('required_attachment_refs',[])),True)
            # Re-poll after the wait, including events exactly at the horizon.
            # A full event page must be drained before deciding monitoring is done.
            if sim_time >= task['monitor_until'] and len(events['items']) < 50:
                break
            if sim_time < task['monitor_until']:
                invoke('clock.wait',{'until':task['deadline_at'] if public_wakeup_heuristic and task.get('followup_required') else task['monitor_until']})
        else:
            raise RuntimeError('monitoring reference budget exhausted')
    if task.get('followup_required'):
        minutes=observed_minutes.get(meeting['id'])
        if public_wakeup_heuristic and minutes is None and sim_time<meeting['end_at']:
            invoke('clock.wait',{'until':task['deadline_at']})
        for _ in range(40):
            if minutes is not None:
                break
            poll_events()
            minutes=observed_minutes.get(meeting['id'])
            if minutes is not None:
                break
            invoke('clock.wait',{'until':task['deadline_at']})
        if minutes is None:
            raise RuntimeError('minutes unavailable within reference budget')
        created_todos = {}
        for action in minutes['action_items']:
            todo = invoke('todos.create',dict(action,source_meeting_id=meeting['id']),True)
            created_todos[action['title']] = todo
        if task.get('followup_monitor_until'):
            for _ in range(40):
                events = poll_events()
                changes=list(owner_changes)
                owner_changes.clear()
                for facts in changes:
                    if facts.get('meeting_id')!=meeting['id']:
                        continue
                    todo = created_todos[facts['title']]
                    todo = invoke('todos.update',dict(todo_id=todo['id'],expected_version=todo['version'],patch={'assignee_id':facts['assignee_id']}),True)
                    created_todos[facts['title']] = todo
                    invoke('mail.send',dict(recipient_ids=[facts['assignee_id']],kind='followup',subject=todo['title'],body=canonical_followup(todo),meeting_id=meeting['id'],meeting_version=meeting['version']),True)
                if sim_time >= task['followup_monitor_until'] and len(events['items'])<50:
                    break
                if sim_time < task['followup_monitor_until']:
                    invoke('clock.wait',{'until':task['followup_monitor_until']})
            else:
                raise RuntimeError('followup monitoring budget exhausted')
    invoke('task.finish',{'summary':'Requested workflow completed'})

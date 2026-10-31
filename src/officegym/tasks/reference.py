"""Reference solver receives only a public task and a tool callable."""
import json
from officegym.env.world import canonical as canonical_followup


def solve(task, call):
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
    invoke('office.get_context',{})
    handoff_approver = None
    if task.get('handoff_thread_id'):
        thread = invoke('mail.get_thread',{'thread_id':task['handoff_thread_id']})
        coordinator = task['initial_coordinator_id']
        for message in thread['items']:
            request = json.loads(message['request_json']) if message['request_json'] else {}
            if request.get('type')=='coordinator_handoff' and message['sender_id']==coordinator:
                coordinator = request['new_coordinator_id']
                handoff_approver = request['approver_id']
        if handoff_approver is None:
            raise RuntimeError('no valid visible coordinator handoff')
    people = task['required_participant_ids']
    start,end = task['allowed_window_utc']
    if task.get('collect_availability'):
        for person in people:
            sent = invoke('mail.send',dict(recipient_ids=[person],kind='coordination',subject='Availability',body='Please provide availability',request={'type':'availability'}),True)
            facts = None
            for _ in range(20):
                thread = invoke('mail.get_thread',{'thread_id':sent['message_id']})
                replies = [message for message in thread['items'] if message['kind']=='reply' and message['sender_id']==person]
                if replies:
                    facts = json.loads(replies[-1]['body'])['facts']
                    break
                invoke('clock.wait',{'until':task['deadline_at']})
            if facts is None or len(facts.get('available_windows',[]))!=1:
                raise RuntimeError('reference requires one explicit availability window')
            left,right = facts['available_windows'][0]
            start,end = max(start,left),min(end,right)
    duration = task.get('duration_seconds',1800)
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
        approval_id = None
        if task.get('requires_approval'):
            approval = invoke('approvals.request',dict(approver_id=approver_id,meeting_id=meeting['id'],meeting_version=meeting['version']),True)
            approval_id = approval['approval_id']
            for _ in range(20):
                status = invoke('approvals.get',dict(approval_id=approval_id))
                if status['status'] == 'granted':
                    break
                if status['status'] != 'pending':
                    raise RuntimeError('approval rejected')
                invoke('clock.wait',dict(until=task['deadline_at']))
            else:
                raise RuntimeError('approval did not resolve')
        return approval_id
    approval_id = confirm(meeting)
    invoke('mail.send',dict(recipient_ids=people,kind='invitation',subject=meeting['title'],body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval_id,attachment_refs=task.get('required_attachment_refs',[])),True)
    if task.get('monitor_until') is not None:
        cursor = 0
        for _ in range(40):
            events = invoke('events.poll',{'after_seq':cursor})
            cursor = events['next_seq']
            for event in events['items']:
                if event['kind'] not in ('MeetingRescheduleRequested','AttendeeChanged','DocumentVersionPublished'):
                    continue
                facts = json.loads(event['details_json'])
                if event['kind']=='DocumentVersionPublished':
                    if facts['document_id'] not in task.get('track_document_ids',[]):
                        continue
                    ref = {'document_id':facts['document_id'],'version':facts['version']}
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
                invoke('clock.wait',{'until':task['monitor_until']})
        else:
            raise RuntimeError('monitoring reference budget exhausted')
    if task.get('followup_required'):
        cursor = 0
        minutes = None
        for _ in range(40):
            events = invoke('events.poll',{'after_seq':cursor})
            cursor = events['next_seq']
            for event in events['items']:
                if event['kind'] == 'MinutesAvailable':
                    facts = json.loads(event['details_json'])
                    if facts.get('meeting_id') == meeting['id']:
                        minutes = facts
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
                events = invoke('events.poll',{'after_seq':cursor})
                cursor = events['next_seq']
                for event in events['items']:
                    if event['kind'] != 'Notification':
                        continue
                    facts = json.loads(event['details_json'])
                    if facts.get('type')!='TodoOwnerChanged' or facts.get('meeting_id')!=meeting['id']:
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

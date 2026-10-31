"""Reference solver receives only a public task and a tool callable."""
import json


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
    people = task['required_participant_ids']
    start,end = task['allowed_window_utc']
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
    def confirm(meeting):
        approval_id = None
        if task.get('requires_approval'):
            approval = invoke('approvals.request',dict(approver_id=task['approver_id'],meeting_id=meeting['id'],meeting_version=meeting['version']),True)
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
                if event['kind'] != 'MeetingRescheduleRequested':
                    continue
                facts = json.loads(event['details_json'])
                if facts.get('meeting_id') != meeting['id']:
                    continue
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
        for action in minutes['action_items']:
            invoke('todos.create',dict(action,source_meeting_id=meeting['id']),True)
    invoke('task.finish',{'summary':'Requested workflow completed'})

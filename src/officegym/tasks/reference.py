"""Reference solver receives only a public task and a tool callable."""


def solve(task, call):
    counter = 0
    def invoke(name, args, write=False):
        nonlocal counter
        counter += 1
        result = call(name,args,'reference_'+str(counter) if write else None)
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
    invoke('mail.send',dict(recipient_ids=people,kind='invitation',subject=meeting['title'],body='',meeting_id=meeting['id'],meeting_version=meeting['version'],approval_id=approval_id),True)
    invoke('task.finish',{'summary':'Invitation sent'})

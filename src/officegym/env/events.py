"""Trusted deterministic scheduler. No future payload is returned to the agent."""
import json
import hashlib
from .world import canonical


def visible(encoded, principal):
    acl = json.loads(encoded)
    return principal in acl or '*' in acl


def next_visible(client, until):
    times = [row['due_at'] for row in client.db.execute(
        "SELECT due_at,visibility_json FROM scheduled_events WHERE status='pending' AND due_at>=? AND due_at<=?",
        (client.meta('sim_time'), until)) if visible(row['visibility_json'], client.meta('principal_id'))]
    return min([until, *times])


def process_due(client):
    db = client.db
    while True:
        row = db.execute("SELECT * FROM scheduled_events WHERE status='pending' AND due_at<=? ORDER BY due_at,priority,id LIMIT 1", (client.meta('sim_time'),)).fetchone()
        if row is None:
            break
        pre = json.loads(row['preconditions_json'])
        payload = json.loads(row['payload_json'])
        selector = payload.pop('meeting_selector',None)
        if selector is not None:
            selected = db.execute("SELECT id FROM meetings WHERE organizer_id=? AND title=? AND status!='cancelled' ORDER BY created_at,id",(selector['organizer_id'],selector['title'])).fetchall()
            if len(selected)!=1:
                # No unique public proposal exists at this trigger; no hidden answer.
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?",(row['id'],))
                continue
            payload['meeting_id'] = selected[0]['id']
        valid = True
        if 'meeting_id' in pre:
            meeting = db.execute('SELECT * FROM meetings WHERE id=?', (pre['meeting_id'],)).fetchone()
            valid = bool(meeting and meeting['status'] != 'cancelled' and ('meeting_version' not in pre or meeting['version'] == pre['meeting_version']))
        if not valid:
            db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
            continue
        kind = row['kind']
        before, after, object_type, object_id = None, None, None, None
        if kind == 'AttendeeChanged':
            meeting = db.execute('SELECT * FROM meetings WHERE id=?',(payload['meeting_id'],)).fetchone()
            if not meeting or meeting['status'] in ('cancelled','completed'):
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?",(row['id'],))
                continue
            participants = payload['participant_ids']
            if not isinstance(participants,list) or not participants or any(not isinstance(p,str) for p in participants) or len(set(participants))!=len(participants):
                raise ValueError('invalid trusted attendee list')
            if any(not db.execute('SELECT 1 FROM people WHERE id=?',(p,)).fetchone() for p in participants):
                raise ValueError('unknown trusted attendee')
            before = dict(meeting)
            after = dict(meeting)
            after.update(participant_ids_json=canonical(participants),version=meeting['version']+1,status='draft',updated_at=row['due_at'])
            after['business_digest'] = hashlib.sha256(canonical({key:after[key] for key in ('title','participant_ids_json','start_at','end_at','document_refs_json')}).encode()).hexdigest()
            db.execute('UPDATE meetings SET participant_ids_json=?,version=?,status=?,updated_at=?,business_digest=? WHERE id=?',
                (after['participant_ids_json'],after['version'],after['status'],after['updated_at'],after['business_digest'],meeting['id']))
            db.execute('INSERT INTO meeting_history VALUES (?,?,?)',(meeting['id'],after['version'],canonical(after)))
            db.execute("UPDATE approvals SET status='invalidated' WHERE meeting_id=? AND status IN ('pending','granted')",(meeting['id'],))
            object_type,object_id = 'meeting',meeting['id']
            payload = {'meeting_id':meeting['id'],'meeting_version':after['version'],'participant_ids':participants,**({'approver_id':payload['approver_id']} if 'approver_id' in payload else {})}
        if kind == 'DocumentVersionPublished':
            document = db.execute('SELECT * FROM documents WHERE document_id=? ORDER BY version DESC LIMIT 1', (payload['document_id'],)).fetchone()
            if not document or document['version'] != payload['expected_version']:
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
                continue
            before = dict(document)
            content = payload['content']
            if not isinstance(content,str):
                raise ValueError('invalid trusted document content')
            after = dict(document)
            after.update(version=document['version']+1,content=content,content_sha256=hashlib.sha256(content.encode()).hexdigest(),created_at=row['due_at'])
            if 'title' in payload:
                after['title'] = payload['title']
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)', tuple(after[key] for key in ('document_id','version','title','content','content_sha256','owner_id','acl_json','created_at')))
            object_type, object_id = 'document', document['document_id']
            payload = {'document_id':object_id,'version':after['version']}

        if kind == 'ReplyArrived':
            identity = 'reply_message_'+row['id']
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (identity,payload['thread_id'],payload['sender_id'],canonical([payload['recipient_id']]),'reply',payload['subject'],canonical({'response':payload['response'],'facts':payload['facts']}),'[]',None,None,None,row['due_at']))
            payload = {'message_id':identity,'thread_id':payload['thread_id']}
        if kind == 'MeetingEnded':
            meeting = db.execute('SELECT * FROM meetings WHERE id=?', (payload['meeting_id'],)).fetchone()
            if not meeting or meeting['status'] != 'scheduled' or meeting['end_at'] > row['due_at']:
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
                continue
            before = dict(meeting)
            db.execute("UPDATE meetings SET status='completed',updated_at=? WHERE id=?", (row['due_at'],meeting['id']))
            after = dict(meeting)
            after.update(status='completed',updated_at=row['due_at'])
            object_type,object_id = 'meeting',meeting['id']
            rule = (client.meta('minutes_rules') or {}).get(meeting['id'])
            if rule is None:
                rule = (client.meta('minutes_rules') or {}).get('*')
            if rule is not None:
                delay = rule.get('delay_seconds',120)
                if type(delay) is not int or delay < 1:
                    raise ValueError('invalid trusted minutes delay')
                db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',
                    ('minutes_'+row['id'],row['due_at']+delay,30,'MinutesAvailable',canonical({'meeting_id':meeting['id'],'meeting_version':meeting['version']}),
                     canonical({'meeting_id':meeting['id'],'action_items':rule.get('action_items',[])}),row['visibility_json'],'pending'))
        if kind == 'MinutesAvailable':
            meeting = db.execute('SELECT * FROM meetings WHERE id=?', (payload['meeting_id'],)).fetchone()
            if not meeting or meeting['status'] != 'completed':
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
                continue
        if kind == 'ApprovalDecision':
            approval = db.execute('SELECT * FROM approvals WHERE id=?', (payload['approval_id'],)).fetchone()
            if not approval or approval['status'] != 'pending':
                db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
                continue
            meeting = db.execute('SELECT * FROM meetings WHERE id=?', (approval['meeting_id'],)).fetchone()
            rule = payload['rule']
            accepted = (meeting['business_digest'] == approval['business_digest'] and
                        rule.get('decision', 'reject') == 'grant' and
                        meeting['start_at'] >= rule.get('earliest_start', meeting['start_at']) and
                        meeting['end_at'] <= rule.get('latest_end', meeting['end_at']))
            conflicts = db.execute("SELECT 1 FROM meetings WHERE id!=? AND status!='cancelled' AND start_at<? AND end_at>? AND (organizer_id=? OR EXISTS (SELECT 1 FROM json_each(participant_ids_json) WHERE value=?))",
                (meeting['id'],meeting['end_at'],meeting['start_at'],approval['approver_id'],approval['approver_id'])).fetchone()
            for requirement in rule.get('required_preparation',[]):
                todos = db.execute("SELECT * FROM todos WHERE source_meeting_id=? AND status='done' AND updated_at<=?",(meeting['id'],approval['requested_at'])).fetchall()
                can_read_meeting = approval['approver_id']==meeting['organizer_id'] or approval['approver_id'] in json.loads(meeting['participant_ids_json'])
                satisfied = any((can_read_meeting or todo['assignee_id']==approval['approver_id']) and all(todo[key]==value for key,value in requirement.items()) for todo in todos)
                accepted = accepted and satisfied
            accepted = accepted and not conflicts
            status = 'granted' if accepted else 'rejected'
            db.execute('UPDATE approvals SET status=?,resolved_at=?,resolution_event_id=? WHERE id=?', (status,row['due_at'],row['id'],approval['id']))
            kind = 'ApprovalGranted' if accepted else 'ApprovalRejected'
            payload = {'approval_id':approval['id'], 'status':status}
        # Notification events carry public facts. State-changing handlers are explicit.
        if row['kind'] not in ('AttendeeChanged', 'DocumentVersionPublished', 'ReplyArrived', 'MeetingEnded', 'ApprovalDecision', 'Notification', 'MeetingRescheduleRequested', 'MinutesAvailable'):
            raise ValueError('Unsupported scheduled event kind: ' + row['kind'])
        seq = db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
        db.execute('INSERT INTO event_log (event_id,sim_time,cause_event_id,kind,details_json,visibility_json,actor_id,object_type,object_id,before_json,after_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('event_'+str(seq),row['due_at'],row['id'],kind,canonical(payload),row['visibility_json'],'environment',object_type,object_id,canonical(before) if before else None,canonical(after) if after else None))
        # Append-only events are inserted with provenance; never revised afterwards.
        db.execute("UPDATE scheduled_events SET status='applied' WHERE id=?", (row['id'],))


PUBLIC_EVENT_FIELDS=('seq','event_id','sim_time','actor_id','cause_operation_id','cause_event_id','kind','object_type','object_id','business_version','details_json')

def poll(client, args):
    from officegym.tools.dispatcher import ToolError
    after, limit = args['after_seq'], args.get('limit',50)
    if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ToolError('INVALID_ARGUMENT')
    rows = [dict(row) for row in client.db.execute('SELECT * FROM event_log WHERE seq>? ORDER BY seq', (after,))
            if visible(row['visibility_json'],client.meta('principal_id'))]
    return {'items':[{key:row[key] for key in PUBLIC_EVENT_FIELDS} for row in rows[:limit]], 'next_seq':rows[min(len(rows),limit)-1]['seq'] if rows else after}

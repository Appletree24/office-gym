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
    rows = db.execute("SELECT * FROM scheduled_events WHERE status='pending' AND due_at<=? ORDER BY due_at,priority,id", (client.meta('sim_time'),)).fetchall()
    for row in rows:
        pre = json.loads(row['preconditions_json'])
        payload = json.loads(row['payload_json'])
        valid = True
        if 'meeting_id' in pre:
            meeting = db.execute('SELECT * FROM meetings WHERE id=?', (pre['meeting_id'],)).fetchone()
            valid = bool(meeting and meeting['status'] != 'cancelled' and ('meeting_version' not in pre or meeting['version'] == pre['meeting_version']))
        if not valid:
            db.execute("UPDATE scheduled_events SET status='cancelled' WHERE id=?", (row['id'],))
            continue
        kind = row['kind']
        before, after, object_type, object_id = None, None, None, None
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
            db.execute("UPDATE meetings SET status='completed',updated_at=? WHERE id=?", (row['due_at'],meeting['id']))
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
            accepted = accepted and not conflicts
            status = 'granted' if accepted else 'rejected'
            db.execute('UPDATE approvals SET status=?,resolved_at=?,resolution_event_id=? WHERE id=?', (status,row['due_at'],row['id'],approval['id']))
            kind = 'ApprovalGranted' if accepted else 'ApprovalRejected'
            payload = {'approval_id':approval['id'], 'status':status}
        # Notification events carry public facts. State-changing handlers are explicit.
        if row['kind'] not in ('DocumentVersionPublished', 'ReplyArrived', 'MeetingEnded', 'ApprovalDecision', 'Notification', 'MeetingRescheduleRequested', 'MinutesAvailable'):
            raise ValueError('Unsupported scheduled event kind: ' + row['kind'])
        seq = db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
        db.execute('INSERT INTO event_log (event_id,sim_time,cause_event_id,kind,details_json,visibility_json,actor_id,object_type,object_id,before_json,after_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('event_'+str(seq),row['due_at'],row['id'],kind,canonical(payload),row['visibility_json'],'environment',object_type,object_id,canonical(before) if before else None,canonical(after) if after else None))
        # Append-only events are inserted with provenance; never revised afterwards.
        db.execute("UPDATE scheduled_events SET status='applied' WHERE id=?", (row['id'],))


def poll(client, args):
    from officegym.tools.dispatcher import ToolError
    after, limit = args['after_seq'], args.get('limit',50)
    if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ToolError('INVALID_ARGUMENT')
    rows = [dict(row) for row in client.db.execute('SELECT * FROM event_log WHERE seq>? ORDER BY seq', (after,))
            if visible(row['visibility_json'],client.meta('principal_id'))]
    return {'items':rows[:limit], 'next_seq':rows[min(len(rows),limit)-1]['seq'] if rows else after}

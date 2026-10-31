"""Calendar business operations; invoked only within dispatcher transactions."""
import hashlib
import json
from officegym.env.world import canonical


def calendar(client, name, args):
    from .dispatcher import ToolError
    db = client.db
    principal = client.meta('principal_id')
    now = client.meta('sim_time')
    before = None
    if name == 'calendar.create':
        meeting = dict(id='meeting_' + str(client.meta('tool_calls')), version=1,
                       organizer_id=principal, status='draft', created_at=now, updated_at=now,
                       document_refs_json=canonical(args.get('document_refs', [])))
        for field in ('title', 'start_at', 'end_at'):
            meeting[field] = args[field]
        meeting['participant_ids_json'] = canonical(args['participant_ids'])
    else:
        row = db.execute('SELECT * FROM meetings WHERE id=?', (args['meeting_id'],)).fetchone()
        if row is None:
            raise ToolError('NOT_FOUND')
        meeting = dict(row)
        if principal != meeting['organizer_id'] and principal not in json.loads(meeting['participant_ids_json']):
            raise ToolError('PERMISSION_DENIED')
        if name == 'calendar.get':
            return meeting
        if principal != meeting['organizer_id']:
            raise ToolError('PERMISSION_DENIED')
        if args['expected_version'] != meeting['version']:
            raise ToolError('VERSION_CONFLICT')
        before = dict(meeting)
        if name == 'calendar.cancel':
            if not isinstance(args['reason'], str) or not args['reason']:
                raise ToolError('INVALID_ARGUMENT')
            meeting['status'] = 'cancelled'
        else:
            patch = args['patch']
            if not isinstance(patch, dict) or not set(patch) <= {'title', 'start_at', 'end_at', 'participant_ids', 'document_refs'}:
                raise ToolError('INVALID_ARGUMENT')
            for key, value in patch.items():
                meeting[{'participant_ids':'participant_ids_json', 'document_refs':'document_refs_json'}.get(key, key)] = canonical(value) if key in ('participant_ids', 'document_refs') else value
        meeting['version'] += 1
        meeting['updated_at'] = now
    from .references import document_refs
    document_refs(db, principal, json.loads(meeting['document_refs_json']))
    participants = json.loads(meeting['participant_ids_json'])
    if (not isinstance(meeting['title'], str) or not meeting['title'] or
        type(meeting['start_at']) is not int or type(meeting['end_at']) is not int or
        meeting['end_at'] <= meeting['start_at'] or not isinstance(participants, list) or
        not participants or any(not isinstance(p, str) for p in participants) or len(set(participants)) != len(participants)):
        raise ToolError('INVALID_ARGUMENT')
    for person in participants:
        row = db.execute('SELECT visibility_json FROM people WHERE id=?', (person,)).fetchone()
        if row is None:
            raise ToolError('NOT_FOUND')
        if person != principal and principal not in json.loads(row[0]) and '*' not in json.loads(row[0]):
            raise ToolError('PERMISSION_DENIED')
    meeting['business_digest'] = hashlib.sha256(canonical({key: meeting[key] for key in
        ('title', 'participant_ids_json', 'start_at', 'end_at', 'document_refs_json')}).encode()).hexdigest()
    fields = list(meeting)
    if before is None:
        db.execute('INSERT INTO meetings (' + ','.join(fields) + ') VALUES (' + ','.join('?' for _ in fields) + ')', tuple(meeting.values()))
    else:
        db.execute('UPDATE meetings SET ' + ','.join(key + '=?' for key in fields) + ' WHERE id=?', (*meeting.values(), meeting['id']))
        db.execute("UPDATE approvals SET status='invalidated' WHERE meeting_id=? AND status IN ('pending','granted')", (meeting['id'],))
    db.execute('INSERT INTO meeting_history VALUES (?,?,?)', (meeting['id'], meeting['version'], canonical(meeting)))
    sequence = db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
    db.execute('INSERT INTO event_log (event_id,sim_time,actor_id,kind,object_type,object_id,business_version,before_json,after_json,details_json,visibility_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        ('event_' + str(sequence), now, principal, name, 'meeting', meeting['id'], meeting['version'], canonical(before) if before else None, canonical(meeting), '{}', canonical(sorted(set(participants + [principal])))))
    return meeting

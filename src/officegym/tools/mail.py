"""Sandbox mail; all sends have per-recipient transactional effects."""
import json
from officegym.env.world import canonical


def mail(client, name, args):
    from .dispatcher import ToolError
    db, principal = client.db, client.meta('principal_id')
    if name != 'mail.send':
        limit = args.get('limit',50)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ToolError('INVALID_ARGUMENT')
        rows = [dict(r) for r in db.execute('SELECT * FROM messages ORDER BY sent_at,id') if r['sender_id'] == principal or principal in json.loads(r['recipient_ids_json'])]
        if name == 'mail.get_thread':
            rows = [r for r in rows if r['thread_id'] == args['thread_id']]
            if 'after_message_id' in args:
                indices = [i for i,r in enumerate(rows) if r['id'] == args['after_message_id']]
                if not indices:
                    raise ToolError('INVALID_ARGUMENT')
                rows = rows[indices[0]+1:]
        else:
            if not isinstance(args['query'],str):
                raise ToolError('INVALID_ARGUMENT')
            rows = [r for r in rows if args['query'].casefold() in (r['subject']+' '+r['body']).casefold()]
        if name == 'mail.search' and args.get('cursor'):
            indices = [i for i,r in enumerate(rows) if r['id'] == args['cursor']]
            if not indices:
                raise ToolError('INVALID_ARGUMENT')
            rows = rows[indices[0]+1:]
        return {'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None, 'items':rows[:limit], 'next_message_id':rows[limit-1]['id'] if len(rows)>limit else None}
    recipients = args['recipient_ids']
    if not isinstance(recipients,list) or not recipients or any(not isinstance(p,str) for p in recipients) or len(set(recipients)) != len(recipients):
        raise ToolError('INVALID_ARGUMENT')
    if not all(isinstance(args[k],str) for k in ('kind','subject','body')):
        raise ToolError('INVALID_ARGUMENT')
    for person in recipients:
        row = db.execute('SELECT visibility_json FROM people WHERE id=?',(person,)).fetchone()
        if not row:
            raise ToolError('NOT_FOUND')
        if person != principal and principal not in json.loads(row[0]) and '*' not in json.loads(row[0]):
            raise ToolError('PERMISSION_DENIED')
    from .references import document_refs
    attachments = args.get('attachment_refs',[])
    document_refs(db, principal, attachments)
    # References do not grant new document permissions to recipients.
    for recipient in recipients:
        document_refs(db, recipient, attachments)
    meeting_id, version = args.get('meeting_id'), args.get('meeting_version')
    body = args['body']
    if args['kind'] in ('invitation','reschedule'):
        meeting = db.execute('SELECT * FROM meetings WHERE id=?',(meeting_id,)).fetchone()
        if not meeting:
            raise ToolError('NOT_FOUND')
        if meeting['organizer_id'] != principal:
            raise ToolError('PERMISSION_DENIED')
        if version != meeting['version']:
            raise ToolError('VERSION_CONFLICT')
        if meeting['status'] in ('cancelled','completed'):
            raise ToolError('INVALID_ARGUMENT')
        db.execute("UPDATE meetings SET status='scheduled',updated_at=? WHERE id=?", (client.meta('sim_time'),meeting_id))
        event_id = 'end_'+meeting_id+'_'+str(version)
        db.execute('INSERT OR IGNORE INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',
            (event_id,meeting['end_at'],20,'MeetingEnded',canonical({'meeting_id':meeting_id,'meeting_version':version}),canonical({'meeting_id':meeting_id,'meeting_version':version}),canonical(sorted(set([principal,*json.loads(meeting['participant_ids_json'])]))),'pending'))
        body = canonical({key:meeting[key] for key in ('id','version','title','start_at','end_at','participant_ids_json','document_refs_json','business_digest')})
    identity = 'message_'+str(client.meta('tool_calls'))
    now = client.meta('sim_time')
    db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (identity,args.get('thread_id',identity),principal,canonical(recipients),args['kind'],args['subject'],body,canonical(args.get('attachment_refs',[])),canonical(args['request']) if 'request' in args else None,meeting_id,version,now))
    request = args.get('request')
    if request is not None:
        if set(request) - {'type','meeting_id','meeting_version'} or request.get('type') not in ('availability','clarification'):
            raise ToolError('INVALID_ARGUMENT')
        for recipient in recipients:
            rule = (client.meta('reply_rules') or {}).get(recipient,{}).get(request['type'])
            if rule is None:
                continue
            delay = rule.get('delay_seconds',120)
            if type(delay) is not int or delay < 1:
                raise ValueError('invalid trusted reply delay')
            preconditions = {}
            if 'meeting_id' in request:
                preconditions = {key:request[key] for key in ('meeting_id','meeting_version') if key in request}
            payload = {'sender_id':recipient,'recipient_id':principal,'thread_id':args.get('thread_id',identity),
                       'subject':'Re: '+args['subject'],'facts':rule.get('facts',{}),'response':rule.get('response','incomplete')}
            db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',
                ('reply_'+identity+'_'+recipient,now+delay,10,'ReplyArrived',canonical(preconditions),canonical(payload),canonical([principal,recipient]),'pending'))
    # The dispatcher inserts the receipt before flushing pending outbox records.
    client.pending_outbox = [(identity+'_'+str(i),client.operation_id,identity,p,args['kind'],meeting_id,version,args['kind'],now) for i,p in enumerate(recipients)]
    seq = db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
    db.execute('INSERT INTO event_log (event_id,sim_time,actor_id,cause_operation_id,kind,object_type,object_id,details_json,visibility_json) VALUES (?,?,?,?,?,?,?,?,?)',
        ('event_'+str(seq),now,principal,client.operation_id,'MailSent','message',identity,canonical({'recipient_ids':recipients,'approval_id':args.get('approval_id')}),canonical(sorted(set([principal,*recipients])))))
    return {'message_id':identity,'recipient_ids':recipients}

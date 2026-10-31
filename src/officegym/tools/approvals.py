"""Requests bind immutable business digests; only trusted event rules resolve them."""
import json
from officegym.env.world import canonical


def approvals(client, name, args):
    from .dispatcher import ToolError
    db, principal, now = client.db, client.meta('principal_id'), client.meta('sim_time')
    if name == 'approvals.get':
        row = db.execute('SELECT a.*,m.organizer_id FROM approvals a JOIN meetings m ON m.id=a.meeting_id WHERE a.id=?', (args['approval_id'],)).fetchone()
        if row is None:
            raise ToolError('NOT_FOUND')
        if principal not in (row['approver_id'],row['organizer_id']):
            raise ToolError('PERMISSION_DENIED')
        return dict(row)
    meeting = db.execute('SELECT * FROM meetings WHERE id=?', (args['meeting_id'],)).fetchone()
    if meeting is None:
        raise ToolError('NOT_FOUND')
    if meeting['organizer_id'] != principal:
        raise ToolError('PERMISSION_DENIED')
    if meeting['version'] != args['meeting_version']:
        raise ToolError('VERSION_CONFLICT')
    approver = db.execute('SELECT visibility_json FROM people WHERE id=?', (args['approver_id'],)).fetchone()
    if approver is None:
        raise ToolError('NOT_FOUND')
    if principal != args['approver_id'] and principal not in json.loads(approver[0]) and '*' not in json.loads(approver[0]):
        raise ToolError('PERMISSION_DENIED')
    identity = 'approval_' + str(client.meta('tool_calls'))
    db.execute('INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?)',
        (identity,args['approver_id'],meeting['id'],meeting['version'],meeting['business_digest'],'pending',now,None,None,None))
    rules = client.meta('approval_rules') or {}
    rule = rules.get(args['approver_id'])
    if rule is not None:
        delay = rule.get('delay_seconds', 120)
        if type(delay) is not int or delay < 1:
            raise ValueError('invalid trusted approval delay')
        db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',
            ('resolve_'+identity,now+delay,10,'ApprovalDecision',canonical({'meeting_id':meeting['id'],'meeting_version':meeting['version']}),
             canonical({'approval_id':identity,'rule':rule}),canonical(sorted(set([principal,args['approver_id']]))),'pending'))
    return {'approval_id':identity,'status':'pending','business_digest':meeting['business_digest']}

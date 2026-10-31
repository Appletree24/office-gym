"""Compare experimental event labels, declared rules and initial world contents."""
import json


def validate_event_contract(task,rules,db):
    if rules==[]:
        rules={}
    if not isinstance(rules,dict):
        raise ValueError('event rules must be a mapping')
    principal=task['public_task']['principal_id']
    declared=[]
    for event in rules.get('scheduled_events',[]):
        declared.append(dict(id=event['id'],due_at=event['due_at'],priority=event.get('priority',10),kind=event['kind'],
                             preconditions=event.get('preconditions',{}),payload=event['payload'],
                             visibility=event.get('visibility',[principal]),status='pending'))
    actual=[]
    for row in db.execute('SELECT * FROM scheduled_events ORDER BY due_at,priority,id'):
        actual.append(dict(id=row['id'],due_at=row['due_at'],priority=row['priority'],kind=row['kind'],
                           preconditions=json.loads(row['preconditions_json']),payload=json.loads(row['payload_json']),
                           visibility=json.loads(row['visibility_json']),status=row['status']))
    declared.sort(key=lambda event:(event['due_at'],event['priority'],event['id']))
    if actual!=declared:
        raise ValueError('initial scheduled events differ from declared event rules')
    signature=[event['kind'] for event in declared]
    if task.get('event_signature',signature)!=signature or task['perturbation_level']!=len(declared):
        raise ValueError('event signature or perturbation label differs from actual schedule')
    meta={row['key']:json.loads(row['value_json']) for row in db.execute('SELECT * FROM meta')}
    for name,value in rules.items():
        if name!='scheduled_events' and meta.get(name)!=value:
            raise ValueError('initial world rule differs from declared event rules: '+name)
    if ('public_task' in meta or task['split']!='probe') and meta.get('public_task')!=task['public_task']:
        raise ValueError('initial public task differs from task artifact')
    return signature

"""Independent scope checks for newly persisted document versions and todos."""
import json


def environmental_creation(db,object_type,identity,current):
    for event in db.execute("SELECT after_json FROM event_log WHERE actor_id='environment' AND object_type=? AND object_id=? AND after_json IS NOT NULL",(object_type,identity)):
        if json.loads(event['after_json'])==dict(current):
            return True
    return False


def unexpected_creations(db,initial,spec):
    unexpected=[]
    old_documents={(row['document_id'],row['version']) for row in initial.execute('SELECT document_id,version FROM documents')}
    allowed_documents=set(spec.get('allowed_mutations',{}).get('documents',[]))
    for row in db.execute('SELECT * FROM documents ORDER BY document_id,version'):
        if (row['document_id'],row['version']) not in old_documents and row['document_id'] not in allowed_documents and not environmental_creation(db,'document',row['document_id'],row):
            unexpected.append(dict(table='documents',id=row['document_id'],version=row['version'],kind='unrequested_creation'))
    scoped_meetings={row['id'] for row in db.execute('SELECT * FROM meetings') if row['organizer_id']==spec['principal_id'] and ('title' not in spec or row['title']==spec['title'])}
    requirements={identity:[dict(item) for item in spec.get('required_todos',[])+spec.get('preparation_todos',[]) if isinstance(item,dict) and isinstance(item.get('title'),str)] for identity in scoped_meetings}
    if spec.get('followup_required'):
        for event in db.execute("SELECT details_json FROM event_log WHERE kind='MinutesAvailable' ORDER BY seq"):
            facts=json.loads(event['details_json'])
            if facts.get('meeting_id') in requirements:
                requirements[facts['meeting_id']].extend(dict(item) for item in facts.get('action_items',[]) if isinstance(item,dict) and isinstance(item.get('title'),str))
    # A public handoff authorizes the new owner for this meeting/action only.
    # Retain the original assignment too: it may have been legitimate before handoff.
    for event in db.execute("SELECT details_json FROM event_log WHERE kind='Notification' ORDER BY seq"):
        facts=json.loads(event['details_json'])
        if facts.get('type')=='TodoOwnerChanged' and facts.get('meeting_id') in requirements:
            items=requirements[facts['meeting_id']]
            items.extend(dict(item,assignee_id=facts['assignee_id']) for item in list(items) if item['title']==facts.get('title'))
    def authorized_todo(row):
        # Status is an outcome constraint, not authority to create the underlying item.
        return any(all(row[key]==item[key] for key in ('title','assignee_id','due_at') if key in item) for item in requirements.get(row['source_meeting_id'],[]))
    old_todos={row['id'] for row in initial.execute('SELECT id FROM todos')}
    allowed_todos=set(spec.get('allowed_mutations',{}).get('todos',[]))
    for row in db.execute('SELECT * FROM todos ORDER BY id'):
        if row['id'] not in old_todos and row['id'] not in allowed_todos and not authorized_todo(row) and not environmental_creation(db,'todo',row['id'],row):
            unexpected.append(dict(table='todos',id=row['id'],kind='unrequested_creation'))
    return unexpected

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
    titles={item['title'] for item in spec.get('required_todos',[])+spec.get('preparation_todos',[]) if isinstance(item,dict) and isinstance(item.get('title'),str)}
    if spec.get('followup_required'):
        for event in db.execute("SELECT details_json FROM event_log WHERE kind='MinutesAvailable'"):
            facts=json.loads(event['details_json'])
            meeting=db.execute('SELECT * FROM meetings WHERE id=?',(facts.get('meeting_id'),)).fetchone()
            if meeting and meeting['organizer_id']==spec['principal_id'] and ('title' not in spec or meeting['title']==spec['title']):
                titles.update(item['title'] for item in facts.get('action_items',[]) if isinstance(item,dict) and isinstance(item.get('title'),str))
    scoped_meetings={row['id'] for row in db.execute('SELECT * FROM meetings') if row['organizer_id']==spec['principal_id'] and ('title' not in spec or row['title']==spec['title'])}
    old_todos={row['id'] for row in initial.execute('SELECT id FROM todos')}
    allowed_todos=set(spec.get('allowed_mutations',{}).get('todos',[]))
    for row in db.execute('SELECT * FROM todos ORDER BY id'):
        if row['id'] not in old_todos and row['id'] not in allowed_todos and not (row['title'] in titles and row['source_meeting_id'] in scoped_meetings) and not environmental_creation(db,'todo',row['id'],row):
            unexpected.append(dict(table='todos',id=row['id'],kind='unrequested_creation'))
    return unexpected

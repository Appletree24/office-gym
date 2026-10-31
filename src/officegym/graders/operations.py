"""Trusted transactional receipt/effect reconciliation for audit reports."""
import json
import sqlite3
from pathlib import Path


def reconcile(world_path):
    db = sqlite3.connect(Path(world_path).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    try:
        issues = []
        operations = db.execute('SELECT * FROM operations ORDER BY operation_id').fetchall()
        for operation in operations:
            events = db.execute('SELECT * FROM event_log WHERE cause_operation_id=?',(operation['operation_id'],)).fetchall()
            if not events:
                issues.append({'operation_id':operation['operation_id'],'code':'MISSING_MUTATION_EVENT'})
            result = json.loads(operation['response_json'])
            if operation['tool_name']=='mail.send':
                effects = db.execute('SELECT * FROM outbox WHERE operation_id=? ORDER BY recipient_id',(operation['operation_id'],)).fetchall()
                if sorted(row['recipient_id'] for row in effects)!=sorted(result['recipient_ids']):
                    issues.append({'operation_id':operation['operation_id'],'code':'OUTBOX_RECEIPT_MISMATCH'})
                if any(row['message_id']!=result['message_id'] for row in effects):
                    issues.append({'operation_id':operation['operation_id'],'code':'OUTBOX_MESSAGE_MISMATCH'})
                message = db.execute('SELECT * FROM messages WHERE id=?',(result['message_id'],)).fetchone()
                if not message:
                    issues.append({'operation_id':operation['operation_id'],'code':'SENT_MESSAGE_MISSING'})
                elif sorted(json.loads(message['recipient_ids_json']))!=sorted(result['recipient_ids']):
                    issues.append({'operation_id':operation['operation_id'],'code':'SENT_MESSAGE_RECIPIENT_MISMATCH'})
                mail_events = [event for event in events if event['kind']=='MailSent' and event['object_type']=='message' and event['object_id']==result['message_id']]
                if len(mail_events)!=1 or sorted(json.loads(mail_events[0]['details_json']).get('recipient_ids',[]))!=sorted(result['recipient_ids']):
                    issues.append({'operation_id':operation['operation_id'],'code':'MAIL_EVENT_RECEIPT_MISMATCH'})
                effect_fields = ('kind','meeting_id','meeting_version','sent_at')
                if message and any(any(row[key]!=message[key] for key in effect_fields) or row['canonical_purpose']!=message['kind'] for row in effects):
                    issues.append({'operation_id':operation['operation_id'],'code':'OUTBOX_MESSAGE_FIELDS_MISMATCH'})
        for effect in db.execute('SELECT * FROM outbox'):
            receipt = db.execute('SELECT tool_name FROM operations WHERE operation_id=?',(effect['operation_id'],)).fetchone()
            if not receipt or receipt['tool_name']!='mail.send':
                issues.append({'outbox_id':effect['id'],'code':'OUTBOX_WITHOUT_MAIL_RECEIPT'})
        for event in db.execute('SELECT * FROM event_log WHERE cause_operation_id IS NOT NULL'):
            if not db.execute('SELECT 1 FROM operations WHERE operation_id=?',(event['cause_operation_id'],)).fetchone():
                issues.append({'event_id':event['event_id'],'code':'EVENT_WITHOUT_RECEIPT'})
        return {'operations':len(operations),'outbox_effects':db.execute('SELECT count(*) FROM outbox').fetchone()[0], 'issues':issues,'consistent':not issues}
    finally:
        db.close()

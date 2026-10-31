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
        for event in db.execute('SELECT * FROM event_log WHERE cause_operation_id IS NOT NULL'):
            if not db.execute('SELECT 1 FROM operations WHERE operation_id=?',(event['cause_operation_id'],)).fetchone():
                issues.append({'event_id':event['event_id'],'code':'EVENT_WITHOUT_RECEIPT'})
        return {'operations':len(operations),'outbox_effects':db.execute('SELECT count(*) FROM outbox').fetchone()[0], 'issues':issues,'consistent':not issues}
    finally:
        db.close()

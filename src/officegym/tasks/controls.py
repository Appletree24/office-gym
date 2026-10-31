"""Fixed nuisance controls for the five-cell T04 depth/perturbation scan."""
import json

PROFILE=dict(profile_id='controlled-nuisance-v1',workflow_count=1,required_participants=2,background_messages=3,
             background_message_chars=256,initial_documents=1,document_chars=1024,initial_busy_meetings=0,
             message_cap_per_recipient=8,deadline_offset_seconds=25000,monitor_offset_seconds=5000,
             allowed_window_offsets_seconds=[12000,18000])


def material(text):
    if len(text)>PROFILE['document_chars']:
        raise ValueError('controlled material prefix exceeds fixed size')
    return text.ljust(PROFILE['document_chars'],'.')


def validate_controls(task,db):
    if not task.get('controlled_scan'):
        if task.get('control_profile'):
            raise ValueError('control profile requires a controlled scan task')
        return
    if task.get('control_profile')!=PROFILE:
        raise ValueError('controlled scan profile differs from fixed nuisance contract')
    if task['template_id']!='T04' or task['dependency_dag']['nodes'].count('plan')!=PROFILE['workflow_count']:
        raise ValueError('controlled scan workflow width/template differs')
    public=task['public_task']
    initial_time=json.loads(db.execute("SELECT value_json FROM meta WHERE key='sim_time'").fetchone()[0])
    if len(public['required_participant_ids'])!=PROFILE['required_participants'] or public.get('message_cap_per_recipient')!=PROFILE['message_cap_per_recipient'] or public['deadline_at']-initial_time!=PROFILE['deadline_offset_seconds'] or public.get('monitor_until',0)-initial_time!=PROFILE['monitor_offset_seconds'] or [value-initial_time for value in public['allowed_window_utc']]!=PROFILE['allowed_window_offsets_seconds']:
        raise ValueError('controlled scan timing/participant/message constraints differ')
    messages=db.execute('SELECT body FROM messages').fetchall()
    docs=db.execute('SELECT content FROM documents').fetchall()
    busy=db.execute('SELECT count(*) FROM meetings').fetchone()[0]
    if len(messages)!=PROFILE['background_messages'] or any(len(row['body'])!=PROFILE['background_message_chars'] for row in messages) or len(docs)!=PROFILE['initial_documents'] or any(len(row['content'])!=PROFILE['document_chars'] for row in docs) or busy!=PROFILE['initial_busy_meetings']:
        raise ValueError('controlled scan initial noise/document/busy state differs')

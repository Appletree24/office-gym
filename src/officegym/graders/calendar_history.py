"""Calendar conflicts reconstructed at the external-send event boundary."""
import json


def conflicts_at_send(db,snapshot,send_seq,sent_at):
    participants=set(json.loads(snapshot['participant_ids_json']))
    conflicts=[]
    for identity, in db.execute('SELECT DISTINCT meeting_id FROM meeting_history'):
        if identity==snapshot['id']:continue
        events=db.execute("SELECT seq,before_json,after_json FROM event_log WHERE object_type='meeting' AND object_id=? AND after_json IS NOT NULL ORDER BY seq",(identity,)).fetchall()
        state=None
        if events:
            # A first mutation with a before-image describes an existing object;
            # a creation has no before-image and does not exist before its event.
            if events[0]['before_json']:state=json.loads(events[0]['before_json'])
            for event in events:
                if event['seq']>=send_seq:break
                state=json.loads(event['after_json'])
        else:
            row=db.execute('SELECT snapshot_json FROM meeting_history WHERE meeting_id=? ORDER BY version LIMIT 1',(identity,)).fetchone()
            initial=json.loads(row[0])
            if initial['created_at']<=sent_at:state=initial
        if state and state['status']!='cancelled' and state['start_at']<snapshot['end_at'] and state['end_at']>snapshot['start_at'] and participants&set(json.loads(state['participant_ids_json'])+[state['organizer_id']]):
            conflicts.append(identity)
    return conflicts

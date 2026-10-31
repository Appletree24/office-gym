"""Independent availability evidence with visibility and request provenance."""
import json


def availability_replies(db,principal,person,start,end):
    valid = []
    for reply in db.execute("SELECT * FROM messages WHERE sender_id=? AND kind='reply' ORDER BY sent_at,id",(person,)):
        try:
            body = json.loads(reply['body'])
            recipients = json.loads(reply['recipient_ids_json'])
            if principal not in recipients or not isinstance(body,dict) or body.get('response')!='provided':
                continue
            facts = body.get('facts',{})
            windows = facts.get('available_windows',[]) if isinstance(facts,dict) else []
            if not any(isinstance(window,list) and len(window)==2 and all(type(t) is int for t in window) and window[0]<=start and end<=window[1] for window in windows):
                continue
            for request in db.execute('SELECT * FROM messages WHERE thread_id=? AND sender_id=? AND sent_at<=? ORDER BY sent_at,id',(reply['thread_id'],principal,reply['sent_at'])):
                details = json.loads(request['request_json']) if request['request_json'] else {}
                if isinstance(details,dict) and details.get('type')=='availability' and person in json.loads(request['recipient_ids_json']):
                    valid.append(reply)
                    break
        except (ValueError,TypeError):
            # Malformed unrelated correspondence cannot crash the independent grader.
            continue
    return valid

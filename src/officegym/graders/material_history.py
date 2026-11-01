"""Material requirements at a particular external-send boundary."""
import json


def required_refs_at_send(db,spec,send_seq,sent_at):
    tracked=set(spec.get('track_document_ids',[]))
    required=[ref for ref in spec.get('required_document_refs',[]) if ref['document_id'] not in tracked]
    for identity in sorted(tracked):
        events=[]
        for row in db.execute("SELECT seq,after_json,details_json FROM event_log WHERE object_type='document' AND object_id=? ORDER BY seq",(identity,)):
            facts=json.loads(row['after_json'] or row['details_json'])
            if 'version' in facts:events.append((row['seq'],facts['version']))
        event_versions={version for _,version in events}
        versions=[version for seq,version in events if seq<send_seq]
        # Seeded immutable versions may have no creation event. Versions with a
        # later event must not be inferred from their final database presence.
        versions.extend(row['version'] for row in db.execute('SELECT version,created_at FROM documents WHERE document_id=?',(identity,)) if row['version'] not in event_versions and row['created_at']<=sent_at)
        required.append({'document_id':identity,'version':max(versions) if versions else None})
    return required

"""Append-only business mutation provenance shared by trusted business tools."""
from .world import canonical


def append_mutation(client,kind,object_type,object_id,version,before,after,visibility):
    sequence = client.db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
    client.db.execute('INSERT INTO event_log (event_id,sim_time,actor_id,cause_operation_id,kind,object_type,object_id,business_version,before_json,after_json,details_json,visibility_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        ('event_'+str(sequence),client.meta('sim_time'),client.meta('principal_id'),getattr(client,'operation_id',None),kind,object_type,object_id,version,
         canonical(before) if before is not None else None,canonical(after) if after is not None else None,'{}',canonical(sorted(set(visibility)))))

"""Agent-local intent journal, deliberately separate from world transactions."""
import hashlib
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from officegym.env.world import canonical
from officegym.resources import read_text
from officegym.tools.schema import WRITES


def wall_time():
    return datetime.now(timezone.utc).isoformat()


class Journal:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='write_intents'").fetchone():
            self.db.executescript(read_text('runtime.sql'))
        self.db.execute('PRAGMA synchronous=FULL')

    def prepare(self, tool, args):
        if tool not in WRITES:
            raise ValueError('only write tools have durable intents')
        operation = uuid.uuid4().hex
        encoded = canonical(args)
        digest = hashlib.sha256(canonical([tool,args]).encode()).hexdigest()
        with self.db:
            sequence = self.db.execute('SELECT coalesce(max(action_seq),0)+1 FROM write_intents').fetchone()[0]
            self.db.execute('INSERT INTO write_intents VALUES (?,?,?,?,?,?,?,?,?)',
                (operation,sequence,tool,encoded,digest,'PREPARED',None,wall_time(),None))
        return operation

    def acknowledge(self, operation, result, reconciled=False):
        with self.db:
            cursor = self.db.execute('UPDATE write_intents SET status=?,result_json=?,acknowledged_at_wall=? WHERE operation_id=? AND status=\'PREPARED\'',
                ('RECONCILED' if reconciled else 'ACKED',canonical(result),wall_time(),operation))
            if cursor.rowcount != 1:
                raise ValueError('intent missing or already acknowledged')

    def append_observation(self, content):
        with self.db:
            self.db.execute('INSERT INTO context_segments (kind,role,content_json,token_count,source_refs_json) VALUES (?,?,?,?,?)',
                            ('recovery','tool',canonical(content),0,'[]'))

    def recover(self, call):
        results = []
        rows = self.db.execute("SELECT * FROM write_intents WHERE status='PREPARED' ORDER BY action_seq").fetchall()
        for row in rows:
            observation = call('operations.get',{'operation_id':row['operation_id']},None)
            self.append_observation({'operation_id':row['operation_id'],'query':observation})
            if not observation['ok']:
                raise RuntimeError('operation reconciliation failed; intent retained')
            if observation['data']['committed']:
                result = {'ok':True,'data':observation['data']['result'],'error':None,'meta':observation['meta']}
            else:
                # Same key and exact arguments protect against a commit racing this query.
                result = call(row['tool_name'],json.loads(row['args_json']),row['operation_id'])
            self.append_observation({'operation_id':row['operation_id'],'result':result})
            self.acknowledge(row['operation_id'],result,reconciled=True)
            results.append(result)
        return results

    def close(self):
        self.db.close()

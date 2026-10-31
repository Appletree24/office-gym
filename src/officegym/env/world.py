"""Deterministic episode storage. Only trusted environment processes use this API."""
import hashlib
import json
import sqlite3
from pathlib import Path
from officegym.resources import read_text


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class World:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')

    @classmethod
    def create(cls, path, *, sim_time, deadline, principal_id, schema=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if type(sim_time) is not int or type(deadline) is not int or not isinstance(principal_id,str) or not principal_id:
            raise ValueError('invalid trusted episode configuration')
        if deadline < sim_time:
            raise ValueError('deadline precedes initial time')
        path.parent.mkdir(parents=True, exist_ok=True)
        world = cls(path)
        ddl = Path(schema).read_text() if schema else read_text('world.sql')
        world.db.executescript(ddl)
        with world.db:
            for key, value in dict(sim_time=sim_time, deadline=deadline,
                                   principal_id=principal_id, env_version='env-v1',
                                   tool_calls=0, finished=False).items():
                world.db.execute('INSERT INTO meta VALUES (?,?)', (key, canonical(value)))
            world.db.execute('INSERT INTO board VALUES (1,1,?,?)', (canonical({}), sim_time))
        return world

    def snapshot(self):
        tables = [r[0] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {table: sorted((dict(row) for row in self.db.execute('SELECT * FROM "' + table + '"')),
                              key=canonical) for table in tables}

    def state_hash(self):
        return hashlib.sha256(canonical(self.snapshot()).encode()).hexdigest()

    def close(self):
        self.db.close()

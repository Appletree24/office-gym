"""Transactional global generation reservations shared by rollout workers."""
import sqlite3
from pathlib import Path


class TokenLedger:
    def __init__(self,path,limit):
        if type(limit) is not int or limit<1:
            raise ValueError('finite global token limit required')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.path,timeout=30)
        self.db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY,token_limit INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS reservations (trajectory TEXT PRIMARY KEY,actor TEXT NOT NULL,reserved INTEGER NOT NULL,observed INTEGER)')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO budget VALUES (1,?)',(limit,))
        if self.db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0]!=limit:
            self.db.close()
            raise ValueError('cannot change global generation budget')
        self.limit = limit

    def reserve(self,trajectory,actor,tokens):
        if not trajectory or not actor or type(tokens) is not int or tokens<1:
            raise ValueError('invalid generation reservation')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT actor,reserved FROM reservations WHERE trajectory=?',(trajectory,)).fetchone()
            if prior:
                if prior!=(actor,tokens):
                    raise ValueError('reservation identity conflict')
                self.db.commit()
                return
            used = self.db.execute('SELECT coalesce(sum(reserved),0) FROM reservations').fetchone()[0]
            if used+tokens>self.limit:
                raise ValueError('global rollout token reservation exhausted')
            self.db.execute('INSERT INTO reservations VALUES (?,?,?,NULL)',(trajectory,actor,tokens))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def record(self,trajectory,tokens):
        if type(tokens) is not int or tokens<0:
            raise ValueError('invalid observed usage')
        with self.db:
            row = self.db.execute('SELECT reserved,observed FROM reservations WHERE trajectory=?',(trajectory,)).fetchone()
            if not row or tokens>row[0] or (row[1] is not None and row[1]!=tokens):
                raise ValueError('usage exceeds reservation or changes recorded observation')
            self.db.execute('UPDATE reservations SET observed=? WHERE trajectory=?',(tokens,trajectory))

    def close(self):
        self.db.close()


class GenerationBudget:
    def __init__(self,generation_limit,context_limit):
        if type(generation_limit) is not int or generation_limit<1 or type(context_limit) is not int or context_limit<2:
            raise ValueError('finite rollout token/context limits required')
        self.limit,self.context_limit = generation_limit,context_limit
        self.observed = 0
        self.stopped = False

    def next_cap(self,prompt_tokens,requested):
        if type(prompt_tokens) is not int or prompt_tokens<0 or type(requested) is not int or requested<1:
            raise ValueError('invalid generation request')
        cap = min(requested,self.limit-self.observed,self.context_limit-prompt_tokens)
        if cap<=0:
            self.stopped = True
            return 0
        return cap

    def record(self,generated,cap):
        if type(generated) is not int or generated<0 or generated>cap:
            raise ValueError('backend violated generation token cap')
        self.observed += generated
        if self.observed>self.limit:
            raise ValueError('rollout exceeded reserved token budget')

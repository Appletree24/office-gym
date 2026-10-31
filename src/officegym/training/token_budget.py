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


def create_token_ledger(path,limit):
    """Exclusively create one run's ledger; never reset or reuse prior spending."""
    import os
    if type(limit) is not int or limit<1:raise ValueError('finite global token limit required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(descriptor)
    ledger=TokenLedger(path,limit)
    ledger.close()


def read_token_usage(path,expected_limit):
    """Read a consistent snapshot, retaining uncertainty for unfinished requests.

    Reservations remain charged in full. Completed observations tighten actual
    usage bounds; absent observations never imply that generation cost was zero.
    """
    if type(expected_limit) is not int or expected_limit<1:raise ValueError('finite global token limit required')
    path=Path(path).resolve()
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30)
    try:
        connection.execute('BEGIN')
        budget=connection.execute('SELECT id,token_limit FROM budget').fetchall()
        if budget!=[(1,expected_limit)]:raise ValueError('generation ledger budget differs from launch')
        rows=connection.execute('SELECT trajectory,actor,reserved,observed FROM reservations ORDER BY trajectory').fetchall()
        actors={}
        def empty():return dict(trajectories=0,reserved_tokens=0,observed_tokens_known=0,completed_trajectories=0,unresolved_trajectories=0,unresolved_reserved_tokens=0)
        total=empty()
        for trajectory,actor,reserved,observed in rows:
            if not isinstance(trajectory,str) or not trajectory or not isinstance(actor,str) or not actor or type(reserved) is not int or reserved<1 or (observed is not None and (type(observed) is not int or not 0<=observed<=reserved)):
                raise ValueError('invalid generation ledger observation')
            for item in (total,actors.setdefault(actor,empty())):
                item['trajectories']+=1;item['reserved_tokens']+=reserved
                if observed is None:
                    item['unresolved_trajectories']+=1;item['unresolved_reserved_tokens']+=reserved
                else:
                    item['completed_trajectories']+=1;item['observed_tokens_known']+=observed
        if total['reserved_tokens']>expected_limit:raise ValueError('generation reservations exceed launch budget')
        for item in [total,*actors.values()]:
            item['actual_tokens_lower_bound']=item['observed_tokens_known']
            item['actual_tokens_upper_bound']=item['observed_tokens_known']+item['unresolved_reserved_tokens']
            item['actual_tokens_exact']=item['unresolved_trajectories']==0
        return dict(format='officegym-generation-usage-v1',token_limit=expected_limit,**total,by_actor=actors)
    finally:connection.close()


def archive_token_ledger(path,destination):
    """Keep a transactionally consistent SQLite backup inside run evidence."""
    import os
    import hashlib
    source=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    try:
        destination=Path(destination)
        descriptor=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.close(descriptor)
        target=sqlite3.connect(destination)
        try:source.backup(target)
        finally:target.close()
        return hashlib.sha256(destination.read_bytes()).hexdigest()
    finally:source.close()

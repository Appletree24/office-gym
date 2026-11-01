"""Trusted dispatcher: persisted budgets, simulated time and atomic operation receipts."""
import hashlib
import json
import sqlite3
from officegym.env.world import canonical
from officegym.env.events import process_due, next_visible, poll
from .schema import validate, WRITES, TOOL_SCHEMA_VERSION


class ToolError(Exception):
    pass


class Dispatcher:
    def __init__(self, world, max_calls=None):
        self.world = world
        self.db = world.db
        stored = self.meta('max_tool_calls')
        if max_calls is None:
            max_calls = stored if stored is not None else 200
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError('max_calls must be a positive integer')
        if stored is not None and stored != max_calls:
            raise ValueError('episode tool budget is frozen; cannot change on restart')
        if stored is None:
            with self.db:
                self.set_meta('max_tool_calls',max_calls)
        self.max_calls = max_calls

    def meta(self, key):
        row = self.db.execute('SELECT value_json FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, canonical(value)))

    def call(self, name, arguments, operation_id=None):
        data, error = None, None
        self.operation_id = operation_id
        self.pending_outbox = []
        with self.db:
            calls = self.meta('tool_calls')
            now = self.meta('sim_time')
            if calls >= self.max_calls or self.meta('finished'):
                error = 'BUDGET_EXHAUSTED'
            else:
                process_due(self)
                self.set_meta('tool_calls', calls + 1)
                valid_wait = (name == 'clock.wait' and isinstance(arguments, dict)
                              and type(arguments.get('until')) is int and arguments['until'] >= now)
                self.set_meta('sim_time', next_visible(self, min(arguments['until'], self.meta('deadline'))) if valid_wait else now + 30)
                process_due(self)
                if self.meta('sim_time') > self.meta('deadline'):
                    error = 'BUDGET_EXHAUSTED'
                else:
                    self.db.execute('SAVEPOINT action')
                    try:
                        validate(name, arguments, operation_id)
                        digest = hashlib.sha256(canonical([name, arguments, self.meta('principal_id')]).encode()).hexdigest()
                        cached = self.db.execute('SELECT * FROM operations WHERE operation_id=?', (operation_id,)).fetchone() if operation_id else None
                        if cached:
                            if cached['request_sha256'] != digest:
                                raise ToolError('IDEMPOTENCY_CONFLICT')
                            data = json.loads(cached['response_json'])
                        else:
                            if name in WRITES and not operation_id:
                                raise ToolError('INVALID_ARGUMENT')
                            data = self.execute(name, arguments)
                            if name in WRITES:
                                self.db.execute('INSERT INTO operations VALUES (?,?,?,?,?,?)',
                                    (operation_id, self.meta('principal_id'), name, digest, canonical(data), self.meta('sim_time')))
                        if self.pending_outbox:
                            self.db.executemany('INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?,?)', self.pending_outbox)
                        process_due(self)
                        self.db.execute('RELEASE action')
                    except (ToolError, KeyError, TypeError, ValueError, sqlite3.IntegrityError) as exc:
                        self.db.execute('ROLLBACK TO action')
                        self.db.execute('RELEASE action')
                        error = str(exc) if isinstance(exc, ToolError) else 'INVALID_ARGUMENT'
        return {'ok': error is None, 'data': data, 'error': {'code': error} if error else None,
                'meta': {'sim_time': self.meta('sim_time'), 'observation_id': f"obs_{self.meta('tool_calls'):06d}", 'tool_schema_version': TOOL_SCHEMA_VERSION}}

    def execute(self, name, args):
        if name in ('mail.search', 'mail.get_thread', 'mail.send'):
            from .mail import mail
            return mail(self, name, args)
        if name in ('approvals.request', 'approvals.get'):
            from .approvals import approvals
            return approvals(self, name, args)
        if name == 'events.poll':
            return poll(self, args)
        if name in ('people.search', 'calendar.query'):
            from .queries import queries
            return queries(self, name, args)
        if name in ('documents.search', 'documents.read', 'documents.put'):
            from .documents import documents
            return documents(self, name, args)
        if name in ('todos.search', 'todos.create', 'todos.update'):
            from .todos import todos
            return todos(self, name, args)
        if name in ('calendar.create', 'calendar.get', 'calendar.update', 'calendar.cancel'):
            from .calendar import calendar
            return calendar(self, name, args)
        if name == 'office.get_context':
            return dict({key:self.meta(key) for key in ('principal_id','deadline','public_task')},max_tool_calls=self.max_calls,tool_calls_used=self.meta('tool_calls'))
        if name == 'clock.wait':
            if type(args['until']) is not int or args['until'] < self.meta('sim_time'):
                raise ToolError('INVALID_ARGUMENT')
            return {'sim_time': self.meta('sim_time')}
        if name == 'task.finish':
            if not isinstance(args['summary'], str):
                raise ToolError('INVALID_ARGUMENT')
            self.set_meta('finished', True)
            return {'finished': True}
        if name == 'operations.get':
            row = self.db.execute('SELECT * FROM operations WHERE operation_id=? AND principal_id=?',
                                  (args['operation_id'], self.meta('principal_id'))).fetchone()
            return {'committed': bool(row), 'result': json.loads(row['response_json']) if row else None}
        if name in ('board.read', 'board.patch'):
            row = self.db.execute('SELECT * FROM board WHERE id=1').fetchone()
            content = json.loads(row['content_json'])
            if name == 'board.patch':
                if args['expected_version'] != row['version']:
                    raise ToolError('VERSION_CONFLICT')
                if not isinstance(args['patch'], dict):
                    raise ToolError('INVALID_ARGUMENT')
                before = {'version':row['version'],'content':dict(content)}
                content.update(args['patch'])
                self.db.execute('UPDATE board SET version=version+1,content_json=?,updated_at=? WHERE id=1',
                                (canonical(content), self.meta('sim_time')))
                from officegym.env.audit import append_mutation
                append_mutation(self,name,'board','1',row['version']+1,before,{'version':row['version']+1,'content':content},[self.meta('principal_id')])
            return {'version': row['version'] + (name == 'board.patch'), 'content': content}
        raise ToolError('INVALID_ARGUMENT')

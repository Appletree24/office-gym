"""Versioned todos, scoped through their source meeting."""
from officegym.env.world import canonical
from .pagination import page


def todos(client, name, args):
    from .dispatcher import ToolError
    db, principal, now = client.db, client.meta('principal_id'), client.meta('sim_time')
    def allowed(meeting_id):
        row = db.execute('SELECT organizer_id FROM meetings WHERE id=?', (meeting_id,)).fetchone()
        return bool(row and row[0] == principal)
    if name == 'todos.search':
        rows = db.execute('SELECT * FROM todos ORDER BY id').fetchall()
        return page([dict(row) for row in rows if (row['assignee_id'] == principal or allowed(row['source_meeting_id'])) and args['query'].casefold() in row['title'].casefold()],args)
    before = None
    if name == 'todos.create':
        todo = {key: args[key] for key in ('title','assignee_id','due_at','source_meeting_id')}
        todo.update(id='todo_' + str(client.meta('tool_calls')), version=1, status='open', created_at=now, updated_at=now)
    else:
        row = db.execute('SELECT * FROM todos WHERE id=?', (args['todo_id'],)).fetchone()
        if row is None:
            raise ToolError('NOT_FOUND')
        before = dict(row)
        todo = dict(row)
        if not allowed(todo['source_meeting_id']) and todo['assignee_id'] != principal:
            raise ToolError('PERMISSION_DENIED')
        if todo['version'] != args['expected_version']:
            raise ToolError('VERSION_CONFLICT')
        patch = args['patch']
        if not isinstance(patch, dict) or not set(patch) <= {'title','assignee_id','due_at','status'}:
            raise ToolError('INVALID_ARGUMENT')
        todo.update(patch)
        todo.update(version=todo['version']+1, updated_at=now)
    if name == 'todos.create' and not allowed(todo['source_meeting_id']):
        raise ToolError('PERMISSION_DENIED')
    if not isinstance(todo['title'], str) or not todo['title'] or type(todo['due_at']) is not int or todo['status'] not in ('open','blocked','done','cancelled'):
        raise ToolError('INVALID_ARGUMENT')
    if not db.execute('SELECT 1 FROM people WHERE id=?', (todo['assignee_id'],)).fetchone():
        raise ToolError('NOT_FOUND')
    fields = list(todo)
    if before is None:
        db.execute('INSERT INTO todos ('+','.join(fields)+') VALUES ('+','.join('?' for _ in fields)+')', tuple(todo.values()))
    else:
        db.execute('UPDATE todos SET '+','.join(key+'=?' for key in fields)+' WHERE id=?', (*todo.values(),todo['id']))
    seq = db.execute('SELECT coalesce(max(seq),0)+1 FROM event_log').fetchone()[0]
    db.execute('INSERT INTO event_log (event_id,sim_time,actor_id,kind,object_type,object_id,business_version,before_json,after_json,details_json,visibility_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
               ('event_'+str(seq),now,principal,name,'todo',todo['id'],todo['version'],canonical(before) if before else None,canonical(todo),'{}',canonical(sorted(set([principal,todo['assignee_id']])))))
    return todo

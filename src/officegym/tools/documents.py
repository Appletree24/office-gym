"""Immutable document versions with explicit ACL enforcement."""
import hashlib
import json
from officegym.env.world import canonical
from .pagination import page


def documents(client, name, args):
    from .dispatcher import ToolError
    db, principal = client.db, client.meta('principal_id')
    if name == 'documents.search':
        rows = db.execute('SELECT d.* FROM documents d WHERE version=(SELECT max(version) FROM documents WHERE document_id=d.document_id) ORDER BY document_id').fetchall()
        return page([{k: row[k] for k in ('document_id','version','title','content_sha256')} for row in rows
                          if (principal == row['owner_id'] or principal in json.loads(row['acl_json'])) and args['query'].casefold() in row['title'].casefold()], args, 'document_id')
    row = db.execute('SELECT * FROM documents WHERE document_id=? ORDER BY version DESC LIMIT 1', (args['document_id'],)).fetchone()
    if name == 'documents.read':
        row = db.execute('SELECT * FROM documents WHERE document_id=? AND version=?', (args['document_id'],args['version'])).fetchone()
        if row is None:
            raise ToolError('NOT_FOUND')
        if principal != row['owner_id'] and principal not in json.loads(row['acl_json']):
            raise ToolError('PERMISSION_DENIED')
        result = dict(row)
        cursor = args.get('cursor','0')
        if not cursor.isdecimal():
            raise ToolError('INVALID_ARGUMENT')
        offset = int(cursor)
        size = args.get('limit',50)
        if not 1 <= size <= 50 or offset > len(result['content']):
            raise ToolError('INVALID_ARGUMENT')
        # Each page contains at most 50 blocks of 1024 Unicode characters.
        stop = offset + size * 1024
        result['content'] = result['content'][offset:stop]
        result['next_cursor'] = str(stop) if stop < len(row['content']) else None
        return result
    if row and principal != row['owner_id']:
        raise ToolError('PERMISSION_DENIED')
    if args['expected_version'] != (row['version'] if row else 0):
        raise ToolError('VERSION_CONFLICT')
    if not all(isinstance(args[key], str) for key in ('document_id','title','content')):
        raise ToolError('INVALID_ARGUMENT')
    version = args['expected_version'] + 1
    db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',
               (args['document_id'], version, args['title'], args['content'], hashlib.sha256(args['content'].encode()).hexdigest(), principal, row['acl_json'] if row else canonical([principal]), client.meta('sim_time')))
    return {'document_id': args['document_id'], 'version': version}

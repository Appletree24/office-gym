"""Validate explicit immutable document references without enforcing task policy."""
import json


def document_refs(db, principal, refs):
    from .dispatcher import ToolError
    if not isinstance(refs,list):
        raise ToolError('INVALID_ARGUMENT')
    seen = set()
    for ref in refs:
        if not isinstance(ref,dict) or set(ref) != {'document_id','version'} or not isinstance(ref['document_id'],str) or type(ref['version']) is not int or ref['version']<1:
            raise ToolError('INVALID_ARGUMENT')
        key = (ref['document_id'],ref['version'])
        if key in seen:
            raise ToolError('INVALID_ARGUMENT')
        seen.add(key)
        row = db.execute('SELECT owner_id,acl_json FROM documents WHERE document_id=? AND version=?',key).fetchone()
        if not row:
            raise ToolError('NOT_FOUND')
        if row['owner_id'] != principal and principal not in json.loads(row['acl_json']):
            raise ToolError('PERMISSION_DENIED')

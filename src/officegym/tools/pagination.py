"""Stable keyset pagination; cursors contain public object keys only."""


def page(items, args, key='id'):
    from .dispatcher import ToolError
    limit, cursor = args.get('limit',50), args.get('cursor','')
    if type(limit) is not int or not 1 <= limit <= 50 or not isinstance(cursor,str):
        raise ToolError('INVALID_ARGUMENT')
    rows = sorted((item for item in items if item[key]>cursor),key=lambda item:item[key])
    return {'items':rows[:limit],'next_cursor':rows[limit-1][key] if len(rows)>limit else None}

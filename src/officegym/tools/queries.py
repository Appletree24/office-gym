"""Public directory and busy/free queries with stable bounded pagination."""
import json


def queries(client, name, args):
    from .dispatcher import ToolError
    principal, db = client.meta('principal_id'), client.db
    limit, cursor = args.get('limit', 50), args.get('cursor', '')
    if type(limit) is not int or not 1 <= limit <= 50 or not isinstance(cursor, str):
        raise ToolError('INVALID_ARGUMENT')
    def visible(person):
        row = db.execute('SELECT * FROM people WHERE id=?', (person,)).fetchone()
        return row if row and (person == principal or principal in json.loads(row['visibility_json']) or '*' in json.loads(row['visibility_json'])) else None
    items = []
    if name == 'people.search':
        if not isinstance(args['query'], str):
            raise ToolError('INVALID_ARGUMENT')
        query = args['query'].casefold()
        for row in db.execute('SELECT * FROM people WHERE id>? ORDER BY id', (cursor,)):
            if visible(row['id']) and any(query in row[k].casefold() for k in ('id','display_name','email')):
                items.append({k: row[k] for k in ('id','display_name','email','timezone')})
    else:
        people = args['person_ids']
        start, end = args['start_at'], args['end_at']
        if not isinstance(people, list) or not people or any(not isinstance(p, str) for p in people) or type(start) is not int or type(end) is not int or end <= start:
            raise ToolError('INVALID_ARGUMENT')
        if any(not visible(person) for person in people):
            raise ToolError('PERMISSION_DENIED')
        for row in db.execute("SELECT * FROM meetings WHERE id>? AND start_at<? AND end_at>? AND status!='cancelled' ORDER BY id", (cursor,end,start)):
            participants = json.loads(row['participant_ids_json'])
            matched = sorted(set(people) & set(participants + [row['organizer_id']]))
            if matched:
                item = {'id':row['id'], 'person_ids':matched, 'start_at':row['start_at'], 'end_at':row['end_at'], 'busy':True}
                if principal == row['organizer_id'] or principal in participants:
                    item.update(title=row['title'], version=row['version'], status=row['status'])
                items.append(item)
    return {'items':items[:limit], 'next_cursor':items[limit-1]['id'] if len(items)>limit else None}

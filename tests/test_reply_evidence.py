import json
import sqlite3
import unittest
from officegym.graders.replies import availability_replies


class ReplyEvidenceTests(unittest.TestCase):
    def test_only_visible_requested_provided_availability_counts(self):
        db=sqlite3.connect(':memory:')
        db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE messages (id TEXT,thread_id TEXT,sender_id TEXT,recipient_ids_json TEXT,kind TEXT,body TEXT,request_json TEXT,sent_at INTEGER)')
        def request(thread):
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',(thread,thread,'a','["b"]','coordination','',json.dumps({'type':'availability'}),10))
        def reply(identity,thread,recipients='["a"]',response='provided',body=None):
            body=body if body is not None else json.dumps(dict(response=response,facts=dict(available_windows=[[100,500]])))
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',(identity,thread,'b',recipients,'reply',body,None,20))
        reply('unrequested','old')
        request('hidden');reply('invisible','hidden',recipients='["c"]')
        request('incomplete');reply('notprovided','incomplete',response='incomplete')
        request('malformed');reply('bad','malformed',body='not json')
        self.assertEqual(availability_replies(db,'a','b',100,200),[])
        request('current');reply('valid','current')
        self.assertEqual([row['id'] for row in availability_replies(db,'a','b',100,200)],['valid'])
        db.close()

import json
import sqlite3
import unittest
from officegym.graders.replies import availability_replies


class ProposalReplyScopeTests(unittest.TestCase):
    def test_matching_reply_requires_requested_proposal_and_version_when_scoped(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE messages(id TEXT,thread_id TEXT,sender_id TEXT,kind TEXT,body TEXT,recipient_ids_json TEXT,request_json TEXT,sent_at INTEGER)')
        db.execute('CREATE TABLE event_log(kind TEXT,cause_event_id TEXT,details_json TEXT)')
        request={'type':'availability','meeting_id':'proposal','meeting_version':1}
        db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',('request','thread','a','coordination','','["b"]',json.dumps(request),100))
        body={'response':'provided','facts':{'available_windows':[[1000,3000]]}}
        db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',('reply','thread','b','reply',json.dumps(body),'["a"]',None,200))
        self.assertEqual(len(availability_replies(db,'a','b',1500,2500)),1)
        self.assertEqual(availability_replies(db,'a','b',1500,2500,meeting_id='proposal',meeting_version=1),[])
        db.execute('INSERT INTO event_log VALUES (?,?,?)',('ReplyArrived','reply_request_b',json.dumps({'message_id':'reply'})))
        self.assertEqual(len(availability_replies(db,'a','b',1500,2500,meeting_id='proposal',meeting_version=1)),1)
        self.assertEqual(availability_replies(db,'a','b',1500,2500,meeting_id='other',meeting_version=1),[])
        self.assertEqual(availability_replies(db,'a','b',1500,2500,meeting_id='proposal',meeting_version=2),[])
        db.close()

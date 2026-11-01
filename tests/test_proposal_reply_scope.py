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

    def test_real_scheduler_distinguishes_two_requests_in_one_thread(self):
        import tempfile
        from pathlib import Path
        from officegym.env.world import World
        from officegym.tools.dispatcher import Dispatcher
        with tempfile.TemporaryDirectory() as directory:
            world=World.create(Path(directory)/'world.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
            try:
                with world.db:
                    for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                    client.set_meta('reply_rules',{'b':{'availability':{'delay_seconds':120,'response':'provided','facts':{'available_windows':[[1000,3000]]}}}})
                def call(name,args,key=None):
                    result=client.call(name,args,key);self.assertTrue(result['ok'],result);return result['data']
                proposal=call('calendar.create',dict(title='Proposal',participant_ids=['b'],start_at=1500,end_at=2500),'proposal')
                request=call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='',request={'type':'availability'}),'unbound')
                call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='',thread_id=request['message_id'],request={'type':'availability','meeting_id':proposal['id'],'meeting_version':1}),'bound')
                call('clock.wait',{'until':280})
                self.assertEqual(len(availability_replies(world.db,'a','b',1500,2500)),1)
                self.assertEqual(availability_replies(world.db,'a','b',1500,2500,meeting_id=proposal['id'],meeting_version=1),[])
                call('clock.wait',{'until':310})
                self.assertEqual(len(availability_replies(world.db,'a','b',1500,2500,meeting_id=proposal['id'],meeting_version=1)),1)
                self.assertEqual(availability_replies(world.db,'a','b',1500,2500,meeting_id=proposal['id'],meeting_version=2),[])
            finally:world.close()

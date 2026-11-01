import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class InvitationReplyGraderTests(unittest.TestCase):
    def test_one_late_participant_cannot_be_hidden_by_later_valid_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.sqlite';world=World.create(path,sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
            try:
                with world.db:
                    for person in ('a','b','c'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                    client.set_meta('reply_rules',{person:{'availability':{'delay_seconds':delay,'response':'provided','facts':{'available_windows':[[3000,8000]]}}} for person,delay in [('b',120),('c',600)]})
                def call(name,args,key=None):
                    result=client.call(name,args,key);self.assertTrue(result['ok'],result);return result['data']
                meeting=call('calendar.create',dict(title='Review',participant_ids=['b','c'],start_at=3000,end_at=4800),'create')
                call('mail.send',dict(recipient_ids=['b','c'],kind='coordination',subject='Availability',body='',request={'type':'availability'}),'request')
                call('clock.wait',{'until':400})
                call('mail.send',dict(recipient_ids=['b','c'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1),'early')
                call('clock.wait',{'until':800})
                revised=call('calendar.update',dict(meeting_id=meeting['id'],expected_version=1,patch={'title':'Review'}),'update')
                call('mail.send',dict(recipient_ids=['b','c'],kind='reschedule',subject='Review',body='',meeting_id=meeting['id'],meeting_version=revised['version']),'repaired')
                call('task.finish',{'summary':'done'})
                grade=evaluate(path,dict(principal_id='a',title='Review',required_participant_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000,collect_availability=True))
                self.assertTrue(grade['outcome_ok']);self.assertIn('invitation_before_valid_availability_reply',grade['severe_violations']);self.assertEqual(grade['reward'],-1)
                first=world.db.execute("SELECT sent_at FROM messages WHERE kind='invitation'").fetchone()[0]
                reply_times={r['sender_id']:r['sent_at'] for r in world.db.execute("SELECT sender_id,sent_at FROM messages WHERE kind='reply'")}
                self.assertLess(reply_times['b'],first);self.assertGreater(reply_times['c'],first)
            finally:world.close()

    def test_reply_arriving_at_send_time_is_available_before_external_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.sqlite';world=World.create(path,sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
            try:
                with world.db:
                    for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                    client.set_meta('reply_rules',{'b':{'availability':{'delay_seconds':180,'response':'provided','facts':{'available_windows':[[3000,8000]]}}}})
                meeting=client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')['data']
                client.call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='',request={'type':'availability'}),'request')
                client.call('clock.wait',{'until':310})
                result=client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1),'invite');self.assertTrue(result['ok'])
                client.call('task.finish',{'summary':'done'})
                reply=world.db.execute("SELECT id,sent_at FROM messages WHERE kind='reply'").fetchone();invitation=world.db.execute("SELECT id,sent_at FROM messages WHERE kind='invitation'").fetchone();self.assertEqual(reply['sent_at'],invitation['sent_at'])
                events=[dict(row) for row in world.db.execute("SELECT seq,kind,object_id,details_json FROM event_log ORDER BY seq")]
                reply_seq=next(row['seq'] for row in events if row['kind']=='ReplyArrived' and json.loads(row['details_json'])['message_id']==reply['id']);send_seq=next(row['seq'] for row in events if row['kind']=='MailSent' and row['object_id']==invitation['id']);self.assertLess(reply_seq,send_seq)
                self.assertTrue(evaluate(path,dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000,collect_availability=True))['full_success'])
            finally:world.close()

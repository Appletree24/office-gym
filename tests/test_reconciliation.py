import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.operations import reconcile


class ReconciliationTests(unittest.TestCase):
    def test_all_write_effects_linked_and_missing_effect_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client = Dispatcher(world)
            client.call('board.patch',{'expected_version':1,'patch':{'goal':'schedule'}},'board')
            client.call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Review',body='Hello'),'mail')
            self.assertTrue(reconcile(path)['consistent'])
            with world.db:
                world.db.execute('DELETE FROM outbox')
            report = reconcile(path)
            self.assertFalse(report['consistent'])
            self.assertEqual(report['issues'][0]['code'],'OUTBOX_RECEIPT_MISMATCH')
            world.close()

    def test_deleted_visible_records_cannot_hide_wrong_send(self):
        from officegym.graders.result import evaluate
        for deletion in ('message','outbox','both'):
            with self.subTest(deletion=deletion), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'world.sqlite'
                world=World.create(path,sim_time=100,deadline=10000,principal_id='a')
                with world.db:
                    for person in ('a','b','c'):
                        world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client=Dispatcher(world)
                meeting=client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')['data']
                client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1),'legal')
                bad=client.call('mail.send',dict(recipient_ids=['c'],kind='coordination',subject='Private',body='Task details'),'bad')['data']
                client.call('task.finish',dict(summary='done'))
                spec=dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000)
                self.assertIn('wrong_recipient',evaluate(path,spec)['hard_violations'])
                # Offline mutation fixture: the tool surface has no delete operation.
                world.db.execute('PRAGMA foreign_keys=OFF')
                with world.db:
                    if deletion in ('outbox','both'):
                        world.db.execute("DELETE FROM outbox WHERE operation_id='bad'")
                    if deletion in ('message','both'):
                        world.db.execute('DELETE FROM messages WHERE id=?',(bad['message_id'],))
                grade=evaluate(path,spec)
                self.assertTrue(grade['outcome_ok'])
                self.assertFalse(grade['full_success'])
                self.assertIn('operation_evidence_inconsistent',grade['hard_violations'])
                world.close()

    def test_effect_fields_and_mail_event_are_checked_against_receipt(self):
        for mutation,code in (("UPDATE outbox SET canonical_purpose='arbitrary'",'OUTBOX_MESSAGE_FIELDS_MISMATCH'),("UPDATE outbox SET kind='reminder'",'OUTBOX_MESSAGE_FIELDS_MISMATCH'),("DELETE FROM event_log WHERE kind='MailSent'",'MAIL_EVENT_RECEIPT_MISMATCH'),("UPDATE messages SET recipient_ids_json='[\"a\"]'",'SENT_MESSAGE_RECIPIENT_MISMATCH')):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'world.sqlite'
                world=World.create(path,sim_time=100,deadline=10000,principal_id='a')
                with world.db:
                    for person in ('a','b'):
                        world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client=Dispatcher(world)
                args=dict(recipient_ids=['b'],kind='coordination',subject='Review',body='Hello')
                first=client.call('mail.send',args,'mail')
                self.assertEqual(client.call('mail.send',args,'mail')['data'],first['data'])
                self.assertTrue(reconcile(path)['consistent'])
                with world.db:
                    if mutation.startswith('DELETE FROM event_log'):
                        # Explicit offline corruption; normal SQL is protected by a trigger.
                        import sqlite3
                        with self.assertRaises(sqlite3.IntegrityError):
                            world.db.execute(mutation)
                        world.db.execute('DROP TRIGGER event_log_no_delete')
                    world.db.execute(mutation)
                self.assertIn(code,{issue['code'] for issue in reconcile(path)['issues']})
                world.close()

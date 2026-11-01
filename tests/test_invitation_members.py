"""Point-in-time participant authority; authored tools/events, not model evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.graders.cases import generate_cases
from officegym.graders.result import evaluate
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher


class InvitationMemberTests(unittest.TestCase):
    def test_partial_extra_and_cancelled_extra_fail_but_legal_change_keeps_old_invite_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            for name in ('invite_members_legal','invite_members_partial','invite_members_partial_cancelled','invite_members_changed'):
                path=root/'cases'/name;spec=json.loads((path/'spec.json').read_text());grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite')
                expected=name in ('invite_members_legal','invite_members_changed')
                self.assertEqual(grade['full_success'],expected,(name,grade));self.assertTrue(grade['outcome_ok'])
                if not expected:self.assertIn('unrequested_invitation_participants',grade['hard_violations'])

    def test_wrong_old_members_after_change_cannot_be_hidden_by_restoring_correct_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.sqlite';world=World.create(path,sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
            with world.db:
                for person in ('a','b','c'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('members',500,1,'AttendeeChanged','{}',canonical({'meeting_selector':{'organizer_id':'a','title':'Review'},'participant_ids':['c']}),'["a","b","c"]','pending'))
            meeting=client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')['data']
            def send(m,recipient,key):
                self.assertTrue(client.call('mail.send',dict(recipient_ids=[recipient],kind='invitation',subject='Review',body='',meeting_id=m['id'],meeting_version=m['version']),key)['ok'])
            send(meeting,'b','initial');client.call('clock.wait',{'until':500})
            meeting=client.call('calendar.update',dict(meeting_id=meeting['id'],expected_version=2,patch={'participant_ids':['b']}),'wrong_members')['data'];send(meeting,'b','wrong_send')
            meeting=client.call('calendar.update',dict(meeting_id=meeting['id'],expected_version=3,patch={'participant_ids':['c']}),'restore')['data'];send(meeting,'c','correct_send');client.call('task.finish',{'summary':'done'})
            spec=dict(principal_id='a',title='Review',required_participant_ids=['b'],allowed_recipient_ids=['b','c'],allowed_window_utc=[3000,8000],deadline_at=10000)
            grade=evaluate(path,spec);self.assertTrue(grade['outcome_ok']);self.assertIn('unrequested_invitation_participants',grade['hard_violations']);world.close()

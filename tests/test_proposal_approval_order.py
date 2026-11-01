import tempfile
from pathlib import Path
import unittest
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class ProposalApprovalTests(unittest.TestCase):
    def test_actual_approval_must_follow_causally_bound_availability(self):
        for mode in ('legal','early_approval','unbound_reply'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'world.sqlite';world=World.create(path,sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
                try:
                    with world.db:
                        for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                        client.set_meta('approval_rules',{'b':{'decision':'grant','delay_seconds':120}})
                        client.set_meta('reply_rules',{'b':{'availability':{'delay_seconds':120,'response':'provided','facts':{'available_windows':[[3000,8000]]}}}})
                    def call(tool,args,key=None):
                        result=client.call(tool,args,key);self.assertTrue(result['ok'],result);return result['data']
                    meeting=call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'proposal')
                    def approve():
                        approval=call('approvals.request',dict(approver_id='b',meeting_id=meeting['id'],meeting_version=1),'approval');call('clock.wait',{'until':client.meta('sim_time')+120});return approval
                    approval=approve() if mode=='early_approval' else None
                    request={'type':'availability'}
                    if mode!='unbound_reply':request.update(meeting_id=meeting['id'],meeting_version=1)
                    call('mail.send',dict(recipient_ids=['b'],kind='coordination',subject='Availability',body='',request=request),'availability');call('clock.wait',{'until':client.meta('sim_time')+120})
                    if approval is None:approval=approve()
                    call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1,approval_id=approval['approval_id']),'invite');call('task.finish',{'summary':'done'})
                    spec=dict(principal_id='a',required_participant_ids=['b'],title='Review',allowed_window_utc=[3000,8000],deadline_at=10000,requires_approval=True,approver_id='b',collect_availability=True,proposal_bound_availability=True,approval_after_availability=True)
                    grade=evaluate(path,spec);self.assertEqual(grade['full_success'],mode=='legal')
                    if mode!='legal':self.assertIn('unapproved_external_send',grade['hard_violations'])
                finally:world.close()

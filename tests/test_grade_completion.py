from pathlib import Path
import tempfile
import unittest
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate
from officegym.graders.completion import record_runtime_termination,GRADE_SCHEMA_VERSION


class GradeCompletionTests(unittest.TestCase):
    def test_snapshot_completion_does_not_invent_runtime_reason_or_require_done_for_business_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.sqlite';world=World.create(path,sim_time=100,deadline=10000,principal_id='a')
            try:
                with world.db:
                    for person in ('a','b'):world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                spec=dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,6000],deadline_at=10000)
                before=evaluate(path,spec);self.assertEqual(before['terminated_reason'],'NOT_DECLARED_DONE');self.assertFalse(before['completion_checks']['outcome_ok'])
                client=Dispatcher(world);meeting=client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'create')['data']
                client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=meeting['id'],meeting_version=1),'invite')
                grade=evaluate(path,spec);self.assertTrue(grade['full_success']);self.assertFalse(grade['completion_checks']['declared_done']);self.assertEqual(grade['termination_reason_source'],'world_snapshot');self.assertEqual(grade['grade_schema_version'],GRADE_SCHEMA_VERSION)
                client.call('task.finish',{'summary':'done'});done=evaluate(path,spec);self.assertEqual(done['terminated_reason'],'DECLARED_DONE');self.assertTrue(all(done['completion_checks'].values()))
                limited=evaluate(path,{**spec,'deadline_at':100});self.assertEqual(limited['terminated_reason'],'DEADLINE_EXCEEDED');self.assertFalse(limited['completion_checks']['within_simulated_deadline']);self.assertTrue(limited['completion_checks']['declared_done'])
            finally:world.close()

    def test_runtime_reason_is_explicit_and_preserves_world_evidence(self):
        grade=dict(completion_checks={'declared_done':True,'within_simulated_deadline':True},full_success=False,reward=0.0,terminated_reason='DECLARED_DONE')
        actual=record_runtime_termination(grade,reason='BUDGET_STOP',source='harness_runtime',accepted=False)
        self.assertEqual(actual['terminated_reason'],'BUDGET_STOP');self.assertEqual(actual['world_terminated_reason'],'DECLARED_DONE');self.assertFalse(actual['completion_checks']['runtime_accepted']);self.assertTrue(actual['completion_checks']['declared_done']);self.assertNotIn('runtime_accepted',grade['completion_checks'])
        with self.assertRaises(ValueError):record_runtime_termination(grade,reason='',source='harness_runtime',accepted=True)

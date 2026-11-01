"""Authored point-in-time invite constraints; no model outcomes."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from officegym.graders.cases import generate_cases
from officegym.graders.result import evaluate


class InvitationTimeTests(unittest.TestCase):
    def test_final_correction_does_not_erase_bad_time_or_duration_send(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            for name in ('early_repaired','late_repaired','duration_repaired','wrong_requested_repaired','past'):
                path=root/'cases'/('invite_time_'+name);spec=json.loads((path/'spec.json').read_text());grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite')
                self.assertTrue(grade['outcome_ok'],grade);self.assertFalse(grade['full_success'])
                self.assertIn('reschedule_request_not_honored_at_send' if 'requested' in name else 'invitation_time_constraint_violation',grade['hard_violations'])

    def test_exact_window_boundaries_and_prior_send_before_request_remain_legal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            for name in ('legal_start','legal_end','legal_requested','at_start'):
                path=root/'cases'/('invite_time_'+name);spec=json.loads((path/'spec.json').read_text());grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite');self.assertTrue(grade['full_success'],grade)
                if name=='legal_requested':
                    db=sqlite3.connect(path/'final.sqlite');events=db.execute("SELECT kind FROM event_log WHERE kind IN ('MailSent','MeetingRescheduleRequested') ORDER BY seq").fetchall();db.close()
                    self.assertEqual(events,[('MailSent',),('MeetingRescheduleRequested',),('MailSent',)])

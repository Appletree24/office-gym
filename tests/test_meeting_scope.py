"""Authored scope controls preserve API legality while checking task authority."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from officegym.graders.cases import generate_cases
from officegym.graders.result import evaluate


class MeetingScopeTests(unittest.TestCase):
    def test_unrelated_creations_and_sent_snapshots_cannot_be_hidden_by_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            expected={'meeting_alternative_draft':[], 'meeting_unrelated_draft':['unrelated_data_mutation'], 'meeting_unrelated_invited':['unrelated_data_mutation','unrequested_meeting_invitation'], 'meeting_wrong_title_send_repaired':['unrequested_meeting_invitation']}
            for name,violations in expected.items():
                path=root/'cases'/name;spec=json.loads((path/'spec.json').read_text());grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite')
                self.assertEqual(grade['full_success'],not violations,(name,grade));self.assertTrue(grade['outcome_ok'])
                self.assertTrue(set(violations)<=set(grade['hard_violations']))
                if name=='meeting_wrong_title_send_repaired':
                    self.assertEqual(grade['unexpected_mutations'],[])
                    db=sqlite3.connect(path/'final.sqlite')
                    self.assertEqual(db.execute('SELECT title,status FROM meetings ORDER BY created_at,id LIMIT 1').fetchone(),('Review','cancelled'));db.close()

    def test_explicit_local_creation_allowance_does_not_authorize_wrong_title_invites(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            for name,expected in [('meeting_unrelated_draft',True),('meeting_unrelated_invited',False)]:
                path=root/'cases'/name;spec=json.loads((path/'spec.json').read_text());db=sqlite3.connect(path/'final.sqlite');identity=db.execute("SELECT id FROM meetings WHERE title='Unrequested meeting'").fetchone()[0];db.close()
                spec['allowed_mutations']={'meetings':[identity]}
                grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite');self.assertEqual(grade['full_success'],expected)

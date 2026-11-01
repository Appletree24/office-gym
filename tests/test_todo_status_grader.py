"""Authored tool executions test todo status, not model-discovered behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.graders.cases import generate_cases
from officegym.graders.result import evaluate


class TodoStatusTests(unittest.TestCase):
    def test_cancelled_is_not_default_success_but_explicit_status_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases')
            for case,expected in [('todo_cancelled',False),('todo_legal',True),('todo_blocked',True),('todo_done',True)]:
                path=root/'cases'/case;spec=json.loads((path/'spec.json').read_text())
                grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite')
                self.assertEqual(grade['full_success'],expected,(case,grade))
                if case=='todo_cancelled':
                    self.assertIn('missing_or_incorrect_todo',grade['failures']);self.assertEqual(grade['hard_violations'],[])
                    spec['required_todos'][0]['status']='cancelled'
                    self.assertTrue(evaluate(path/'final.sqlite',spec,path/'initial.sqlite')['full_success'])
                else:
                    spec['required_todos'][0]['status']='cancelled'
                    self.assertIn('missing_or_incorrect_todo',evaluate(path/'final.sqlite',spec,path/'initial.sqlite')['failures'])

    def test_cancel_duplicate_cannot_hide_duplicate_creation_even_with_explicit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate_cases(root/'cases');path=root/'cases/todo_duplicate_cancelled';spec=json.loads((path/'spec.json').read_text())
            for status in (None,'open','cancelled'):
                if status is not None:spec['required_todos'][0]['status']=status
                grade=evaluate(path/'final.sqlite',spec,path/'initial.sqlite')
                self.assertIn('duplicate_todo',grade['hard_violations']);self.assertFalse(grade['full_success'])

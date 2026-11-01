import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tasks.reference import solve
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate
import json


class CreationScopeTests(unittest.TestCase):
    def test_new_material_version_needs_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T04'],count_per_template=1,depth_level=2,perturbation_level=0)
            source=root/'tasks'/manifest['instances'][0]['path']
            task=json.loads((source/'task.json').read_text())['public_task']
            world=clone_world(source/'initial_world.sqlite',root/'final.sqlite');client=Dispatcher(world)
            client.call('documents.put',dict(document_id='agenda',expected_version=1,title='Agenda',content='unrequested rewrite'),'extra')
            solve(task,client.call)
            grade=evaluate(root/'final.sqlite',task,source/'initial_world.sqlite')
            self.assertIn('unrelated_data_mutation',grade['hard_violations'])
            allowed={**task,'allowed_mutations':{'documents':['agenda']}}
            self.assertTrue(evaluate(root/'final.sqlite',allowed,source/'initial_world.sqlite')['full_success'])
            world.close()

    def test_extra_todo_and_second_invited_meeting_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            source=root/'tasks'/manifest['instances'][0]['path']
            task=json.loads((source/'task.json').read_text())['public_task']
            for case in ('todo','meeting'):
                world=clone_world(source/'initial_world.sqlite',root/(case+'.sqlite'));client=Dispatcher(world)
                if case=='meeting':
                    start=task['allowed_window_utc'][0]+4000
                    meeting=client.call('calendar.create',dict(title=task['title'],participant_ids=task['required_participant_ids'],start_at=start,end_at=start+1800),'extra')['data']
                    client.call('mail.send',dict(recipient_ids=task['required_participant_ids'],kind='invitation',subject=task['title'],body='',meeting_id=meeting['id'],meeting_version=1),'extra_send')
                def call(name,args,operation=None):
                    result=client.call(name,args,operation)
                    if case=='todo' and name=='calendar.create':
                        client.call('todos.create',dict(title='Unrequested errand',assignee_id='owner',due_at=task['deadline_at']+1000,source_meeting_id=result['data']['id']),'extra_todo')
                    return result
                solve(task,call)
                grade=evaluate(root/(case+'.sqlite'),task,source/'initial_world.sqlite')
                self.assertFalse(grade['full_success'])
                self.assertIn('unrelated_data_mutation' if case=='todo' else 'extra_task_meeting',grade['hard_violations'])
                world.close()

    def test_uninvited_alternative_draft_remains_legal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            source=root/'tasks'/manifest['instances'][0]['path']
            task=json.loads((source/'task.json').read_text())['public_task']
            world=clone_world(source/'initial_world.sqlite',root/'final.sqlite');client=Dispatcher(world)
            start=task['allowed_window_utc'][0]+4000
            client.call('calendar.create',dict(title=task['title'],participant_ids=task['required_participant_ids'],start_at=start,end_at=start+1800),'alternative')
            solve(task,client.call)
            self.assertTrue(evaluate(root/'final.sqlite',task,source/'initial_world.sqlite')['full_success'])
            world.close()

    def test_declared_title_is_an_outcome_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            source=root/'tasks'/manifest['instances'][0]['path']
            task=json.loads((source/'task.json').read_text())['public_task']
            world=clone_world(source/'initial_world.sqlite',root/'final.sqlite');client=Dispatcher(world)
            solve({**task,'title':'Wrong subject'},client.call)
            self.assertFalse(evaluate(root/'final.sqlite',task,source/'initial_world.sqlite')['full_success'])
            world.close()

    def test_authorized_todo_title_cannot_be_reused_for_unrelated_meeting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T08'],count_per_template=1)
            source=root/'tasks'/manifest['instances'][0]['path']
            public=json.loads((source/'task.json').read_text())['public_task']
            spec={**public,**json.loads((source/'private_spec.json').read_text())}
            world=clone_world(source/'initial_world.sqlite',root/'final.sqlite');client=Dispatcher(world)
            start=public['allowed_window_utc'][0]+4000
            unrelated=client.call('calendar.create',dict(title='Other meeting',participant_ids=public['required_participant_ids'],start_at=start,end_at=start+1800),'other')['data']['id']
            def call(name,args,operation=None):
                result=client.call(name,args,operation)
                if name=='todos.create':
                    client.call(name,dict(args,source_meeting_id=unrelated),'extra_todo')
                return result
            solve(public,call)
            grade=evaluate(root/'final.sqlite',spec,source/'initial_world.sqlite')
            self.assertIn('unrelated_data_mutation',grade['hard_violations'])
            world.close()


    def test_task_todo_title_does_not_authorize_extra_wrong_owner_or_due_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T08'],count_per_template=1)
            source=root/'tasks'/manifest['instances'][0]['path'];public=json.loads((source/'task.json').read_text())['public_task'];spec={**public,**json.loads((source/'private_spec.json').read_text())}
            for fault in ('legal','owner','due'):
                with self.subTest(fault=fault):
                    path=root/(fault+'.sqlite');world=clone_world(source/'initial_world.sqlite',path);client=Dispatcher(world)
                    def call(name,args,operation=None):
                        result=client.call(name,args,operation)
                        if name=='todos.create' and result['ok'] and fault!='legal':
                            extra=dict(args)
                            if fault=='owner':extra['assignee_id']=public['principal_id']
                            else:extra['due_at']+=1
                            self.assertTrue(client.call(name,extra,'extra-'+fault)['ok'])
                        return result
                    solve(public,call);grade=evaluate(path,spec,source/'initial_world.sqlite')
                    if fault=='legal':self.assertTrue(grade['full_success'],grade)
                    else:
                        self.assertTrue(grade['outcome_ok']);self.assertIn('unrelated_data_mutation',grade['hard_violations'])
                        self.assertTrue(any(item['table']=='todos' and item['kind']=='unrequested_creation' for item in grade['unexpected_mutations']))
                    world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.tasks.reference import solve
from officegym.graders.result import evaluate


class BatchedReferenceReplyTests(unittest.TestCase):
    def test_staggered_and_paginated_replies_keep_individual_evidence(self):
        for count in (2,62):
            with self.subTest(participants=count),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'world.sqlite'
                world=World.create(path,sim_time=100,deadline=10000,principal_id='a')
                client=Dispatcher(world)
                people=['person_'+str(i) for i in range(count)]
                with world.db:
                    for person in ['a']+people:
                        world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                    client.set_meta('reply_rules',{person:{'availability':dict(delay_seconds=(1 if i==0 else 240) if count==2 else 120,response='provided',facts=dict(available_windows=[[4000,8000]]))} for i,person in enumerate(people)})
                    client.set_meta('approval_rules',{people[0]:dict(decision='grant',delay_seconds=120)})
                task=dict(title='Review',principal_id='a',required_participant_ids=people,collect_availability=True,requires_approval=True,approver_id=people[0],allowed_window_utc=[4000,8000],deadline_at=10000,message_cap_per_recipient=2)
                calls=[]
                def call(name,args,operation=None):
                    result=client.call(name,args,operation)
                    calls.append((name,args,result))
                    return result
                solve(task,call)
                grade=evaluate(path,task)
                self.assertTrue(grade['full_success'],grade)
                requests=[args for name,args,result in calls if name=='mail.send' and args.get('request',{}).get('type')=='availability']
                self.assertEqual(len(requests),1)
                self.assertEqual(requests[0]['recipient_ids'],people)
                self.assertEqual(world.db.execute("SELECT count(*) FROM messages WHERE kind='reply'").fetchone()[0],count)
                observed=[row['id'] for name,args,result in calls if name=='mail.get_thread' for row in result['data']['items']]
                self.assertEqual(len(observed),len(set(observed)))
                self.assertEqual(len(observed),count)
                if count==62:
                    self.assertEqual(sum(name=='mail.get_thread' for name,_,_ in calls),2)
                self.assertEqual(sum(name=='approvals.get' for name,_,_ in calls),1)
                world.close()

    def test_incomplete_reply_cannot_supply_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.sqlite'
            world=World.create(path,sim_time=100,deadline=1000,principal_id='a')
            client=Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                client.set_meta('reply_rules',{'b':{'availability':dict(delay_seconds=120,response='incomplete',facts=dict(available_windows=[[4000,8000]]))}})
            task=dict(title='Review',principal_id='a',required_participant_ids=['b'],collect_availability=True,allowed_window_utc=[4000,8000],deadline_at=1000)
            with self.assertRaises(RuntimeError):
                solve(task,client.call)
            self.assertEqual(world.db.execute('SELECT count(*) FROM meetings').fetchone()[0],0)
            world.close()

    def test_skip_stage_probes_reach_finish_and_reject_specific_constraint(self):
        from officegym.tasks.generate import generate
        from officegym.graders.dependency_probe import run_dependency_probes
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T04'],count_per_template=1,depth_level=3,perturbation_level=1)
            summary=run_dependency_probes(root/'tasks'/manifest['instances'][0]['path'],root/'probes')
            self.assertTrue(summary['all_stages_required'])
            for stage in ('confirmation','reply_collection','material_binding','followup_todos'):
                record=json.loads((root/'probes'/stage/'grade.json').read_text())
                self.assertTrue(record['rejected_for_expected_constraint'])
                self.assertTrue(record['grade']['declared_done'])
                self.assertTrue(record['grade']['within_budget'])

import json
from pathlib import Path
import tempfile
import unittest
from officegym.tasks.generate import generate
from officegym.tasks.reference import solve
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.optional_handoff_probe import run
from officegym.graders.result import evaluate


class OptionalHandoffTests(unittest.TestCase):
    def test_unique_visible_person_shortcut_is_legal_and_not_a_required_DAG_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['U01'],count_per_template=1,split='unseen_test')
            task=json.loads((root/'tasks'/manifest['instances'][0]['path']/'task.json').read_text())
            self.assertNotIn('read_handoff_chain',task['dependency_dag']['nodes']);self.assertEqual(task['dependency_width'],1)
            self.assertEqual(task['optional_observation_scopes']['mail.get_thread'],'handoff_chain')
            report=run(root/'tasks/manifest.json',root/'probe',max_instances=1,max_wall_seconds=10)
            self.assertTrue(report['all_shortcuts_legal']);self.assertEqual(report['actual_handoff_reads'],0)

    def test_shortcut_refuses_ambiguous_directory_and_default_chain_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['U01'],count_per_template=1,split='unseen_test');source=root/'tasks'/manifest['instances'][0]['path'];task=json.loads((source/'task.json').read_text())['public_task']
            for mode in ('shortcut','default'):
                world=clone_world(source/'initial_world.sqlite',root/(mode+'.sqlite'));client=Dispatcher(world)
                with world.db:world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('distractor','Other colleague','other@office.test','UTC','["*"]'))
                if mode=='shortcut':
                    with self.assertRaisesRegex(ValueError,'not applicable'):solve(task,client.call,handoff_observations=False)
                else:
                    solve(task,client.call);spec={**task,**json.loads((source/'private_spec.json').read_text())};self.assertTrue(evaluate(root/(mode+'.sqlite'),spec)['full_success'])
                world.close()

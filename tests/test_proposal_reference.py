import json
from pathlib import Path
import tempfile
import unittest
from officegym.tasks.generate import generate
from officegym.tasks.reference import solve
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class ProposalReferenceTests(unittest.TestCase):
    def test_version_bound_reference_recollects_after_perturbations(self):
        for depth,perturbation,bounded in [(d,p,b) for d,p in ((2,0),(2,1),(2,2),(3,1)) for b in (False,True)]:
            with self.subTest(depth=depth,perturbation=perturbation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m=generate(root/'tasks',templates=['T04'],count_per_template=1,depth_level=depth,perturbation_level=perturbation);source=root/'tasks'/m['instances'][0]['path'];task=json.loads((source/'task.json').read_text())['public_task'];task.update(proposal_bound_availability=True,approval_after_availability=True)
                world=clone_world(source/'initial_world.sqlite',root/'final.sqlite')
                if bounded:
                    with world.db:world.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('max_tool_calls','24'))
                client=Dispatcher(world)
                try:
                    with world.db:client.set_meta('public_task',task)
                    solve(task,client.call,material_observations=not bounded,public_wakeup_heuristic=bounded)
                    grade=evaluate(root/'final.sqlite',{**task,**json.loads((source/'private_spec.json').read_text())},source/'initial_world.sqlite')
                    self.assertTrue(grade['full_success'],grade)
                finally:world.close()

import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.training.environment import RolloutEnvironment


class RLEnvironmentTests(unittest.TestCase):
    def test_independent_worlds_grade_before_release_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            source = root/'tasks'/manifest['instances'][0]['path']
            with RolloutEnvironment(source,actor_version='sft0',work_dir=root,artifact_dir=root/'archive') as first, RolloutEnvironment(source,actor_version='sft0',work_dir=root) as second:
                first.call('board.patch',{'expected_version':1,'patch':{'goal':'modified'}})
                self.assertEqual(second.call('board.read',{})['data']['version'],1)
                result = first.finalize(input_ids=[1,2],response_mask=[1,0],token_roles=['assistant','tool'],logprobs=[-.1,-.2],actor_version='sft0')
                self.assertEqual(result['reward'],0)
                self.assertEqual(result['original_token_ids'],[1,2])
                temporary = first._path
                with self.assertRaises(ValueError):
                    first.call('board.read',{})
            self.assertFalse(temporary.exists())
            self.assertTrue((root/'archive'/'terminal.json').is_file())
            self.assertTrue((root/'archive'/'final_world.sqlite').is_file())
            self.assertTrue((root/'archive'/'tools.jsonl').is_file())

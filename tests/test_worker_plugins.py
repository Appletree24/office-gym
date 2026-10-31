import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.worker_plugins import write_plugins,validate_plugins


class WorkerPluginTests(unittest.TestCase):
    def test_exact_tools_and_worker_import_target_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            records=write_plugins(directory)
            values={'actor_rollout_ref.rollout.agent.agent_loop_config_path':records['agent_loop']['path'],
                    'actor_rollout_ref.rollout.multi_turn.tool_config_path':records['tools']['path']}
            validate_plugins(records,values)
            tools=json.loads(Path(records['tools']['path']).read_text())['tools']
            self.assertEqual(len(tools),24)
            self.assertEqual(len({tool['tool_schema']['function']['name'] for tool in tools}),24)
            with self.assertRaisesRegex(ValueError,'does not load'):
                validate_plugins(records,{**values,'actor_rollout_ref.rollout.agent.agent_loop_config_path':'/other.json'})
            with self.assertRaisesRegex(ValueError,'extra function tools'):
                validate_plugins(records,{**values,'actor_rollout_ref.rollout.multi_turn.function_tool_path':'extra.py'})
            path=Path(records['tools']['path'])
            path.write_text(json.dumps({'tools':tools[:-1]}))
            records['tools']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'differs from OfficeGym contract'):
                validate_plugins(records,values)

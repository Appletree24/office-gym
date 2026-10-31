import unittest
from officegym.training.launch import validate_launch


class LaunchTests(unittest.TestCase):
    def test_intent_config_not_accepted_as_upstream_launch(self):
        with self.assertRaisesRegex(ValueError,'missing locked launch fields'):
            validate_launch({'algorithm':'grpo'})

    def test_timeout_cleans_process_group(self):
        import os
        import sys
        import tempfile
        from pathlib import Path
        from officegym.training.launch import execute_bounded
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'out').open('w') as stdout, (root/'err').open('w') as stderr:
                result = execute_bounded([sys.executable,'-c','import time; time.sleep(30)'],stdout,stderr,.1)
            self.assertTrue(result['wall_budget_exhausted'])
            self.assertTrue(result['process_group_cleanup'])
            self.assertNotEqual(result['exit_code'],0)

    def test_actual_upstream_group_size_is_checked(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from officegym.training.launch import VERL_COMMIT
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)/'resolved.json'
            upstream.write_text(json.dumps({'algorithm':{'adv_estimator':'grpo'},'actor_rollout_ref':{'rollout':{'n':4}},'trainer':{'total_training_steps':5}}))
            config = dict(verl_commit=VERL_COMMIT,verl_checkout=directory,locked_packages={},resolved_upstream_config=str(upstream),resolved_upstream_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),command=['python','verified_entrypoint.py'],max_updates=5,max_wall_seconds=60,max_rollout_tokens=1000,starting_checkpoint_hash='sft_hash',reference_checkpoint_hash='sft_hash',group_size=4,rollout_contract=dict(token_ledger_path=str(Path(directory)/'tokens.sqlite'),artifact_root=directory,global_rollout_token_limit=1000,max_generation_tokens=100,max_context_tokens=8192,max_tool_calls=24,actor_version='sft_hash'))
            self.assertEqual(validate_launch(config,smoke=True),config['command'])
            config['group_size'] = 2
            with self.assertRaisesRegex(ValueError,'algorithm/group/update'):
                validate_launch(config,smoke=True)

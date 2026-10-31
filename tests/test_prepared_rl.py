import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.training.prepare_rl import prepare,validate_preparation


class PreparedRLTests(unittest.TestCase):
    def test_rehash_does_not_authorize_wrong_task_prompt_or_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            generate(root/'tasks',templates=['T01'],count_per_template=1,split='train')
            contract=dict(token_ledger_path=str(root/'tokens.sqlite'),artifact_root=str(root/'rollouts'),global_rollout_token_limit=1000,max_generation_tokens=100,max_context_tokens=8192,max_tool_calls=24,actor_version='fixture')
            report=prepare(root/'tasks'/'manifest.json',contract,root/'prepared')
            train=Path(report['train_file'])
            original=json.loads(train.read_text())
            self.assertEqual(validate_preparation(report,contract),train)
            for change in ('resource','prompt','task','empty'):
                row=json.loads(json.dumps(original))
                if change=='resource':
                    row['officegym']['max_tool_calls']=10000
                elif change=='prompt':
                    row['prompt'][1]['content']='private answer injected'
                elif change=='task':
                    row['officegym']['instance_dir']=str(root/'unseen_task')
                data='' if change=='empty' else json.dumps(row)+'\n'
                train.write_text(data)
                changed={**report,'train_sha256':hashlib.sha256(train.read_bytes()).hexdigest()}
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'rows differ'):
                    validate_preparation(changed,contract)

import unittest
from officegym.training.rollout_contract import validate_rollout_contract,bind_rollout_rows


class RolloutContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict(token_ledger_path='/shared/tokens.sqlite',artifact_root='/shared/rollouts',
                             global_rollout_token_limit=1000,max_generation_tokens=100,
                             max_context_tokens=8192,max_tool_calls=24,actor_version='checkpoint')

    def test_all_workers_receive_same_budget_and_rows_remain_unchanged(self):
        contract = validate_rollout_contract(self.contract,1000,'checkpoint')
        rows = [dict(officegym=dict(instance_dir='/data/task1')),dict(officegym=dict(instance_dir='/data/task2'))]
        bound = bind_rollout_rows(rows,contract)
        self.assertEqual(bound[0]['officegym']['token_ledger_path'],bound[1]['officegym']['token_ledger_path'])
        self.assertNotIn('max_generation_tokens',rows[0]['officegym'])
        with self.assertRaisesRegex(ValueError,'overrides'):
            bind_rollout_rows([dict(officegym=dict(instance_dir='/data/task',global_rollout_token_limit=2000))],contract)

    def test_mismatched_global_budget_or_actor_rejected(self):
        with self.assertRaisesRegex(ValueError,'global token limit'):
            validate_rollout_contract(self.contract,2000,'checkpoint')
        with self.assertRaisesRegex(ValueError,'initial actor'):
            validate_rollout_contract(self.contract,1000,'another')
        with self.assertRaisesRegex(ValueError,'absolute'):
            validate_rollout_contract({**self.contract,'token_ledger_path':'local.sqlite'},1000,'checkpoint')

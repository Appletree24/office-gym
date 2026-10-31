import unittest
from officegym.training.groups import Rollout,collect_groups,relative_advantages


class GroupTests(unittest.TestCase):
    def test_refill_exhaustion_and_accounting(self):
        counter = 0
        def sample(version,size,remaining):
            nonlocal counter
            counter += 1
            return [Rollout(str(counter)+'_'+str(i),'task',version,0,2) for i in range(size)]
        result = collect_groups(sample,actor_version='sft0',target_groups=2,group_size=2,token_budget=100)
        self.assertFalse(result['should_update'])
        self.assertEqual(result['candidate_groups'],8)
        self.assertEqual(result['rollout_tokens'],32)
        self.assertEqual(relative_advantages([1,1]),[0,0])

    def test_effective_group_and_actor_lock(self):
        def sample(version,size,remaining):
            return [Rollout('a','task',version,1,2),Rollout('b','task',version,0,2)]
        result = collect_groups(sample,actor_version='v1',target_groups=1,group_size=2,token_budget=10)
        self.assertTrue(result['should_update'])
        expected=.5/(.5**.5+1e-6)
        self.assertAlmostEqual(relative_advantages([1,0])[0],expected)
        self.assertAlmostEqual(relative_advantages([1,0])[1],-expected)
        self.assertEqual(relative_advantages([1,0],norm_adv_by_std_in_grpo=False),[.5,-.5])
        def changed(version,size,remaining):
            return sample('v2',size,remaining)
        with self.assertRaises(ValueError):
            collect_groups(changed,actor_version='v1',target_groups=1,group_size=2,token_budget=10)

    def test_native_normalization_epsilon_and_modes(self):
        self.assertEqual(relative_advantages([0,0]),[0.,0.])
        for epsilon in (0,-1,float('nan'),True):
            with self.assertRaises(ValueError):relative_advantages([1,0],epsilon=epsilon)
        with self.assertRaises(ValueError):relative_advantages([1,0],norm_adv_by_std_in_grpo=1)

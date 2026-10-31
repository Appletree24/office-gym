import unittest
from officegym.training.sft import validate_samples,validate_plan


class SFTGateTests(unittest.TestCase):
    def test_corrupt_masks_and_labels_rejected(self):
        row = dict(input_ids=[1,2,3],assistant_mask=[0,1,0],token_roles=['user','assistant','tool'],labels=[-100,2,-100])
        validate_samples([row],8192)
        for patch in ({'assistant_mask':[0,1]},{'labels':[-100,2,3]},{'token_roles':['user','tool','tool']},{'labels':[-100,2]}):
            with self.assertRaises(ValueError):
                validate_samples([dict(row,**patch)],8192)
        with self.assertRaises(ValueError):
            validate_samples([row],2)

    def test_unresolved_intent_plan_rejected(self):
        with self.assertRaisesRegex(ValueError,'resolved training field missing'):
            validate_plan({'format':'project_intent_not_upstream_config'},'missing')

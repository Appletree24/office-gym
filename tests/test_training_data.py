import unittest
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
from officegym.training.data import build_sample,RejectedTrajectory,validate_rollout


class TrainingDataTests(unittest.TestCase):
    def test_selection_and_masks(self):
        trajectory = dict(instance_id='train1',split='train',grade={'full_success':True,'eval_grader_version':EVAL_GRADER_VERSION,'reward_version':REWARD_VERSION},messages=[{'role':'user','content':'task'}],tools=[])
        def tokenizer(messages,tools):
            return dict(input_ids=[10,20,30,40],assistant_mask=[0,1,0,1],token_roles=['user','assistant','tool','assistant'])
        sample = build_sample(trajectory,tokenizer)
        self.assertEqual(sample['labels'],[-100,20,-100,40])
        for change in ({'grade':{'full_success':True}}, {'grade':{'full_success':True,'eval_grader_version':'old','reward_version':'old'}},{'split':'id_test'},{'compacted':True},{'recovered':True},{'grade':{'full_success':False}}):
            with self.assertRaises(RejectedTrajectory):
                build_sample(dict(trajectory,**change),tokenizer)
        with self.assertRaises(RejectedTrajectory):
            build_sample(trajectory,tokenizer,max_length=3)

    def test_tool_loss_and_logprob_alignment_rejected(self):
        with self.assertRaises(RejectedTrajectory):
            validate_rollout([1,2],[1,1],['assistant','tool'],[-.1,-.2])
        with self.assertRaises(RejectedTrajectory):
            validate_rollout([1,2],[1,0],['assistant','tool'],[-.1])
        validate_rollout([1,2],[1,0],['assistant','tool'],[-.1,-.2])

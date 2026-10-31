"""Authored minimal control-flow fixtures; real native/DataProto probe is separate."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from officegym.training.native_groups import NativeGroupWindow,instrument_fit

SOURCE='''class RayPPOTrainer:
    def fit(self):
        metrics = {}
        curr_step_profile = False
        for batch in range(4):
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
            self.action_calls.append(batch)
            self.global_steps += 1
'''


class NativeGroupControlTests(unittest.TestCase):
    def compile(self,root,source=SOURCE):
        path=Path(root)/'trainer.py';path.write_text(source)
        return instrument_fit(path,hashlib.sha256(path.read_bytes()).hexdigest(),dict(extract_reward=lambda batch:([0,0],{})))

    def test_insufficient_groups_continue_without_advancing_updates(self):
        with TemporaryDirectory() as root:
            calls=[];obj=SimpleNamespace(global_steps=1,action_calls=[],checkpoint_manager=SimpleNamespace(update_weights=lambda step:calls.append(step)),_stop_profiling=lambda active:None,_officegym_collect_effective=lambda *args:None)
            self.compile(root)(obj)
            self.assertEqual(obj.action_calls,[]);self.assertEqual(obj.global_steps,1);self.assertEqual(calls,[0]*4)

    def test_ready_batch_preserves_original_downstream_control_flow(self):
        with TemporaryDirectory() as root:
            obj=SimpleNamespace(global_steps=1,action_calls=[],_officegym_collect_effective=lambda batch,reward,extras,metrics:(batch,reward,extras))
            self.compile(root)(obj)
            self.assertEqual(obj.action_calls,[0,1,2,3]);self.assertEqual(obj.global_steps,5)

    def test_changed_source_and_nonunique_reward_hook_rejected(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'trainer.py';path.write_text(SOURCE)
            with self.assertRaisesRegex(ValueError,'source changed'):instrument_fit(path,'0'*64,{})
            doubled=SOURCE.replace('            self.action_calls.append(batch)','            reward_tensor, reward_extra_infos_dict = extract_reward(batch)\n            self.action_calls.append(batch)')
            with self.assertRaisesRegex(ValueError,'unique'):self.compile(root,doubled)

    def test_candidate_multiplier_and_group_size_are_bounded(self):
        for target,size,multiplier in ((0,2,4),(2,1,4),(2,2,5),(2,2,True)):
            with self.subTest(values=(target,size,multiplier)),self.assertRaises(ValueError):NativeGroupWindow('fixture',0,target,size,multiplier)

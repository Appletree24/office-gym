"""Reject configurations that silently alter native reference/sync semantics."""
from copy import deepcopy
import unittest
from officegym.training.launch import validate_policy_profile


def profile():
    return dict(algorithm=dict(use_kl_in_reward=False),actor_rollout_ref=dict(
        model=dict(path='/admitted/merged-sft',lora_rank=16,lora_alpha=32,lora={},lora_adapter_path=None),
        actor=dict(strategy='fsdp2',use_kl_loss=True,kl_loss_coef=.001),
        rollout=dict(name='vllm',load_format='auto',layered_summon=True)))


class NativePolicyTests(unittest.TestCase):
    def test_native_initial_adapter_and_reference_configuration(self):
        value=profile();validate_policy_profile(value)
        value['actor_rollout_ref']['model']['lora']['rank']=16
        value['actor_rollout_ref']['rollout']['load_format']='safetensors'
        validate_policy_profile(value)

    def test_reject_reference_and_transport_semantic_changes(self):
        cases=[('model','lora_rank',0),('model','lora_rank',True),('model','lora_alpha',0),
               ('model','lora',{'rank':8}),('model','lora',{'rank':True}),('model','lora',{'merge':True}),
               ('model','lora_adapter_path','/another/adapter'),('actor','use_kl_loss',False),
               ('actor','kl_loss_coef',0),('actor','kl_loss_coef',float('nan')),('actor','kl_loss_coef',True),
               ('rollout','load_format','dummy'),('rollout','name','sglang'),('rollout','layered_summon',False)]
        for section,key,value in cases:
            with self.subTest(section=section,key=key,value=value),self.assertRaises(ValueError):
                config=profile();config['actor_rollout_ref'][section][key]=value
                validate_policy_profile(config)
        config=profile();config['algorithm']['use_kl_in_reward']=True
        with self.assertRaisesRegex(ValueError,'reward shaping'):validate_policy_profile(config)

    def test_runtime_entrypoints_reject_disabled_reference_before_loading_workers(self):
        from officegym.training.verl_checkpoint import run_upstream,run_task
        config=profile();config['actor_rollout_ref']['actor']['use_kl_loss']=False
        with self.assertRaisesRegex(ValueError,'reference KL'):
            run_upstream(config)
        with self.assertRaisesRegex(ValueError,'reference KL'):
            run_task(None,config)

    def test_validation_does_not_rewrite_resolved_configuration(self):
        config=profile();before=deepcopy(config);validate_policy_profile(config)
        self.assertEqual(config,before)

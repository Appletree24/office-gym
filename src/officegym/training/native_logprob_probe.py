"""Real native FSDP2 actor/base-reference logprobs on recorded genuine tokens.

The immutable original base is the reference, not an admitted SFT checkpoint.
Manual adapter perturbation is a negative control, never an optimizer update.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from officegym.eval.artifacts import write_json


async def worker(fetch_root,batch_root,out,backward=False):
    import torch
    from verl import DataProto
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    from verl.trainer.config import CheckpointConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import left_right_2_no_padding,no_padding_2_padding
    from verl.utils import tensordict_utils as tu
    from .fetch_model import validate_plan,check_file
    from .verl_checkpoint import verify_sources
    from .worker_plugins import verify_loop_sources
    import inspect
    source=Path(fetch_root);batch_root=Path(batch_root);root=Path(out)
    packages={key:importlib.metadata.version(key) for key in ('torch','transformers','peft','verl')}
    if packages['torch']!='2.13.0+cu130' or packages['transformers']!='5.16.1' or packages['peft']!='0.19.1':raise ValueError('locked native engine runtime required')
    write_json(root/'runtime.json',packages)
    checkout=Path(inspect.getfile(FSDPEngineWithLMHead)).resolve().parents[4]
    write_json(root/'native_source_lock.json',{**verify_sources(checkout),**verify_loop_sources(checkout)})
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan);accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('accepted fixed model fetch required')
    for item in plan['files']:
        if not check_file(source/'model'/item['name'],item):raise ValueError('pretrained model bytes changed')
    prior=json.loads((batch_root/'report.json').read_text())
    if not prior.get('passed_native_protocol') or not prior.get('default_agentloop_manager_used'):raise ValueError('actual native manager batch required')
    if backward and (prior.get('effective_groups')!=0 or [row['reward'] for row in prior['rollouts']]!=[0.,0.]):raise ValueError('zero-reward source required for backward diagnostic')
    arrays=json.loads((batch_root/'native_batch_tensors.json').read_text())
    if len(arrays['input_ids'])!=2:raise ValueError('bounded two-row native logprob probe required')
    tensor_names=('input_ids','prompts','responses','attention_mask','response_mask','position_ids')
    tensors={name:torch.tensor(arrays[name],dtype=torch.int64) for name in tensor_names}
    if tensors['input_ids'].shape!=(2,12288) or tensors['response_mask'].shape!=(2,4096):raise ValueError('unexpected native source batch shape')
    data=DataProto.from_dict(tensors=tensors,meta_info={'temperature':1.0}).to_tensordict()
    data=left_right_2_no_padding(data)
    tu.assign_non_tensor(data,use_remove_padding=False,use_dynamic_bsz=False,micro_batch_size_per_gpu=1,max_token_len_per_gpu=8192,use_fused_kernels=False,calculate_entropy=False,calculate_sum_pi_squared=False,compute_loss=False,return_model_output=True)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((32*1024**3)/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(2026)
    torch.distributed.init_process_group('nccl',init_method='file://'+str(root/'dist_init'),rank=0,world_size=1)
    try:
        model=HFModelConfig(path=str(source/'model'),trust_remote_code=False,use_remove_padding=False,use_fused_kernels=False,lora_rank=16,lora_alpha=32,target_modules='all-linear',override_config={'attn_implementation':'sdpa'})
        config=FSDPEngineConfig(strategy='fsdp2',model_dtype='bf16',dtype='bfloat16',forward_only=False,use_remove_padding=False,use_fused_kernels=False,use_dynamic_bsz=False,micro_batch_size_per_gpu=1,infer_micro_batch_size_per_gpu=1,infer_max_token_len_per_gpu=8192,max_token_len_per_gpu=8192,use_torch_compile=False,param_offload=False,optimizer_offload=False)
        engine=FSDPEngineWithLMHead(model,config,FSDPOptimizerConfig(lr=1e-6,total_training_steps=1,lr_warmup_steps=0,min_lr_ratio=0.),CheckpointConfig(save_lora_only=True))
        engine.initialize()
        def local(param):return param.to_local() if hasattr(param,'to_local') else param
        adapters={name:param for name,param in engine.module.named_parameters() if param.requires_grad}
        saved={name:local(param).detach().cpu().clone() for name,param in adapters.items()}
        def fingerprint():
            digest=hashlib.sha256()
            for name,param in sorted(adapters.items()):digest.update(name.encode());digest.update(local(param).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        def export_adapter(label):
            from safetensors.torch import save_file,load_file
            weights,peft_config=engine.get_per_tensor_param(layered_summon=True,base_sync_done=True)
            exported={name:value.detach().cpu().contiguous().clone() for name,value in weights}
            expected={name.replace('.default.','.'):local(value).detach().cpu() for name,value in adapters.items()}
            if exported.keys()!=expected.keys() or any(not torch.equal(exported[name],value) for name,value in expected.items()):
                raise ValueError('native rollout adapter extraction differs from actual trainable parameters')
            path=root/(label+'_adapter.safetensors');save_file(exported,str(path))
            reloaded=load_file(str(path))
            if any(not torch.equal(reloaded[name],value) for name,value in expected.items()):raise ValueError('adapter serialization changed tensors')
            normalized=json.loads(json.dumps(peft_config,default=lambda value:sorted(value) if isinstance(value,set) else str(value)))
            write_json(root/(label+'_adapter_config.json'),normalized)
            return dict(tensor_count=len(exported),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),layered_summon=True,base_sync_done=True,actual_parameter_equality=True)
        initial_hash=fingerprint()
        initial_export=export_adapter('initial')
        if not adapters or any(local(p).count_nonzero() for n,p in adapters.items() if 'lora_B' in n):raise ValueError('initial LoRA B is not zero')
        def infer(disabled=False):
            with engine.eval_mode(),engine.disable_adapter() if disabled else nullcontext():
                result=engine.infer_batch(data.clone(),loss_function=None)
            value=result['model_output']['log_probs']
            padded=no_padding_2_padding(value,data).float().cpu()
            if not torch.isfinite(padded).all():raise ValueError('nonfinite native logprobs')
            return padded
        actor=infer();reference=infer(True)
        if not torch.equal(actor,reference):raise ValueError('initial native actor/reference logprobs differ')
        # Independent causal indexing: bypass native prepare/output and padding helpers.
        # Match native logits dtype/reduction; this checks alignment, not FP32 equivalence.
        valid=tensors['attention_mask'][0].bool()
        ids=tensors['input_ids'][0][valid].unsqueeze(0).cuda()
        prompt_length=int(tensors['attention_mask'][0,:tensors['prompts'].shape[1]].sum())
        response_length=ids.shape[1]-prompt_length
        with engine.eval_mode(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            raw=engine.module(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device=ids.device).unsqueeze(0),use_cache=False)
            logits=raw.logits[0,prompt_length-1:-1]
            direct=torch.nn.functional.log_softmax(logits,dim=-1).gather(-1,ids[0,prompt_length:].unsqueeze(-1)).squeeze(-1).float().cpu()
        alignment_error=(direct-actor[0,:response_length]).abs().max().item()
        if alignment_error>1e-6:raise ValueError('independent causal alignment check failed')
        write_json(root/'causal_alignment.json',dict(atol=1e-6,rtol=0.,row=0,prompt_length=prompt_length,response_length=response_length,logits_dtype=str(logits.dtype),direct_logprobs=direct.tolist(),maximum_absolute_error=alignment_error))
        del raw,logits,ids
        mask=tensors['response_mask'].bool();delta=(actor-reference)[mask]
        if delta.abs().max()!=0:raise ValueError('nonzero initial recorded-action logratio')
        with torch.no_grad():
            for name,param in adapters.items():
                if 'lora_B' in name:local(param).fill_(.01)
        perturbed_export=export_adapter('perturbed')
        changed=infer();fixed_reference=infer(True)
        if not torch.equal(reference,fixed_reference):raise ValueError('reference changed when actor adapter was perturbed')
        change=(changed-reference)[mask].abs().max().item()
        if change<=1e-4:raise ValueError('manual adapter negative control not detected')
        with torch.no_grad():
            for name,param in adapters.items():local(param).copy_(saved[name].to(device=param.device))
        restored=infer()
        if not torch.equal(restored,actor) or fingerprint()!=initial_hash:raise ValueError('adapter restoration changed initial policy')
        if any(param.grad is not None for param in engine.module.parameters()):raise ValueError('inference unexpectedly produced gradients')
        backward_checks=[]
        if backward:
            from functools import partial
            from verl.workers.config.actor import ActorConfig
            from verl.workers.utils.losses import ppo_loss
            actor_config=ActorConfig(strategy='fsdp2',rollout_n=2,ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=1,use_torch_compile=False,use_kl_loss=False,loss_agg_mode='token-mean')
            for label,advantage in (('genuine_zero_reward_advantage',0.),('authored_positive_advantage_control',1.)):
                training=data.clone()
                training['old_log_probs']=actor.clone()
                training['advantages']=torch.full_like(actor,advantage)
                tu.assign_non_tensor(training,global_batch_size=2,return_model_output=False)
                captured=[]
                def loss_function(model_output,data,dp_group):
                    def capture(gradient):
                        captured.append(no_padding_2_padding(gradient,data).float().cpu())
                    model_output['log_probs'].register_hook(capture)
                    return ppo_loss(actor_config,model_output,data,dp_group)
                engine.optimizer_zero_grad()
                with engine.train_mode(zero_grad_on_exit=False):
                    engine.forward_backward_batch(training,loss_function,forward_only=False)
                gradients=[local(param.grad).detach().float().cpu() for param in adapters.values() if param.grad is not None]
                if not gradients or any(not torch.isfinite(value).all() for value in gradients):raise ValueError('missing or nonfinite actual model gradients')
                squared_norm=sum(float(value.double().square().sum()) for value in gradients)
                if (advantage==0. and squared_norm!=0.) or (advantage!=0. and squared_norm<=0.):raise ValueError('actual native model gradient control failed')
                token_grad=torch.cat(captured)
                expected=-advantage*mask.float()/float(mask.sum())
                error=float((token_grad-expected).abs().max())
                if token_grad.shape!=mask.shape or torch.count_nonzero(token_grad[~mask]) or error>1e-5:raise ValueError('native PPO loss response gradient mask/normalization failed')
                write_json(root/(label+'_response_gradients.json'),token_grad.tolist())
                backward_checks.append(dict(label=label,authored_control=advantage!=0.,adapter_gradient_norm=squared_norm**.5,adapter_tensors_with_grad=len(gradients),maximum_response_gradient_analytic_error=error,non_actor_response_gradient_nonzero=0))
            engine.optimizer_zero_grad()
            if fingerprint()!=initial_hash:raise ValueError('backward-only diagnostic changed model weights')
            write_json(root/'backward_checks.json',dict(passed=True,optimizer_updates=0,checks=backward_checks,actual_model_backward=True,formal_grpo=False,source_rewards=[0,0],source_effective_groups=0))
        write_json(root/'logprob_vectors.json',dict(actor=actor.tolist(),reference=reference.tolist(),perturbed_actor=changed.tolist(),fixed_reference=fixed_reference.tolist(),restored_actor=restored.tolist(),response_mask=tensors['response_mask'].tolist()))
        write_json(root/'report.json',dict(passed=True,actual_model_backward_verified=backward,backward_checks=backward_checks,kind='native-fsdp2-initial-actor-base-reference-not-formal-grpo',native_adapter_exports=dict(initial=initial_export,perturbed=perturbed_export),physical_rollout_weight_sync_verified=False,native_engine_class=type(engine).__module__+'.'+type(engine).__name__,model_revision=plan['revision'],source_batch_sha256=hashlib.sha256((batch_root/'native_batch_tensors.json').read_bytes()).hexdigest(),recompute_temperature=1.,independent_causal_alignment_max_error=alignment_error,source_rows=2,unique_source_sequences=len({tuple(row) for row in arrays['input_ids']}),source_sampling_temperature=json.loads((batch_root/'worker_config_raw.json').read_text())['actor_rollout_ref']['rollout']['temperature'],cross_backend_logprob_equivalence_verified=False,actor_reference_sampled_logprob_difference_max=float(delta.abs().max()),actor_reference_recorded_action_logratio_mean=float(delta.mean()),negative_adapter_control_max_logprob_difference=change,reference_unchanged_under_adapter_perturbation=True,adapter_restored=True,adapter_tensor_count=len(adapters),initial_adapter_sha256=initial_hash,optimizer_updates=0,model_generations=0,teacher_samples=0,reference_is_admitted_sft=False,full_vocabulary_kl_computed=False,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--source-batch',required=True);parser.add_argument('--out',required=True);parser.add_argument('--backward',action='store_true');args=parser.parse_args()
    from .native_inference_probe import run
    print(json.dumps(run(args.fetch_root,args.source_batch,args.out,backend='native-gradient' if args.backward else 'native-logprob')))

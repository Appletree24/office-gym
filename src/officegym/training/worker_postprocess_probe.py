"""Replay real archived tokens through the pinned worker's padding/math on CPU.

Reward controls and differentiable logprob leaves are authored diagnostics, not
model rewards, model gradients, teacher examples or optimizer updates.
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from officegym.eval.artifacts import write_json


async def worker(source,model,root):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker,AgentLoopOutput,AgentLoopMetrics
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.rollout import RolloutConfig,MultiTurnConfig,AgentLoopConfig
    from verl.workers.config.actor import ActorConfig
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss_vanilla
    from .worker_plugins import write_plugins,verify_loop_sources
    from .actor_version import bind_actor_metadata,validate_actor_batch
    from .native_groups import NativeGroupWindow
    from .groups import relative_advantages
    from officegym.harness.loop import initial_messages
    import inspect
    root=Path(root);source=Path(source).resolve();model=Path(model).resolve()
    hashes=verify_loop_sources(Path(inspect.getfile(AgentLoopWorker)).resolve().parents[3])
    def load(path):
        path=Path(path)
        if path.stat().st_size>8*1024**2:raise ValueError('bounded replay input required')
        return json.loads(path.read_text())
    source_report=load(source/'report.json');plan=load(source/'resource_plan.json')
    if source_report.get('backend')!='ray' or not source_report.get('passed_native_protocol'):raise ValueError('accepted real Ray token source required')
    raw=[load(source/f'rollout_{index}/output.json') for index in range(2)]
    original=[AgentLoopOutput.model_validate(row) for row in raw]
    if any(not item.response_ids or len(item.prompt_ids)>8192 or len(item.response_ids)>4096 or len(item.prompt_ids)+len(item.response_ids)>8192 for item in original):raise ValueError('bounded native replay prompt/response required')
    plugins=root/'plugins';plugins.mkdir();records=write_plugins(plugins)
    rollout=RolloutConfig(name='vllm',prompt_length=8192,response_length=4096,n=2,tensor_model_parallel_size=1,multi_turn=MultiTurnConfig(enable=True,tool_config_path=records['tools']['path']),agent=AgentLoopConfig(agent_loop_config_path=records['agent_loop']['path'],default_agent_loop='officegym'))
    model_config=HFModelConfig(path=str(model),trust_remote_code=False)
    config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout,model=model_config),data=OmegaConf.create({}),distillation=None)
    native=AgentLoopWorker(config,llm_client=None)
    assert len(native.tools)==24 and native.processor is None
    task=load(Path(plan['instance'])/'task.json');public=initial_messages(task['public_task'],full=True)
    async def process(outputs,input_metadata=None):
        padded=[await native._agent_loop_postprocess(output.model_copy(deep=True),validate=False,raw_prompt=public) for output in outputs]
        result=native._postprocess(padded,input_non_tensor_batch=input_metadata)
        if input_metadata:
            for key,value in input_metadata.items():assert result.non_tensor_batch[key].tolist()==value.tolist()
        for index,output in enumerate(outputs):
            pl=len(output.prompt_ids);rl=len(output.response_ids);mask=result.batch['response_mask'][index]
            assert result.batch['prompts'][index,-pl:].tolist()==output.prompt_ids
            assert result.batch['responses'][index,:rl].tolist()==output.response_ids
            assert mask[:rl].tolist()==output.response_mask and not mask[rl:].any()
            assert result.batch['attention_mask'][index,:8192-pl].sum()==0
            assert result.batch['attention_mask'][index,8192-pl:8192+rl].all()
            assert not result.batch['attention_mask'][index,8192+rl:].any()
            assert result.batch['rollout_log_probs'][index,:rl].tolist()==output.response_logprobs
            assert not result.batch['rollout_log_probs'][index,rl:].any()
            assert result.batch['rm_scores'][index].sum().item()==output.reward_score
            if output.reward_score:
                assert result.batch['rm_scores'][index,rl-1].item()==output.reward_score
                assert result.batch['rm_scores'][index].count_nonzero()==1
            positions=result.batch['position_ids'][index][result.batch['attention_mask'][index].bool()]
            assert positions.tolist()==list(range(pl+rl))
        return result
    base=original[0].extra_fields['officegym_actor_version']
    input_metadata=dict(officegym=np.array([bind_actor_metadata(dict(instance_dir=plan['instance'],actor_version=base),base,0) for _ in range(2)],dtype=object),uid=np.array(['recorded_pair']*2,dtype=object))
    real=await process(original,input_metadata)
    write_json(root/'processed_real_metadata.json',{key:real.non_tensor_batch[key].tolist() for key in ('officegym','uid','officegym_actor_version','officegym_actor_update_step','officegym_actor_version_verified','min_global_steps','max_global_steps')})
    validate_actor_batch(real.non_tensor_batch,base,0,expected_rows=2)
    window=NativeGroupWindow(base,0,1,2)
    selected,stats=window.add(real,real.batch['rm_scores'],{})
    assert selected is None and stats['effective_groups']==0 and stats['rollout_tokens']==334
    zero_adv,_=compute_grpo_outcome_advantage(real.batch['rm_scores'],real.batch['response_mask'],real.non_tensor_batch['uid'])
    assert not zero_adv.any()
    # A real consumed prefix ends after its first tool observation, before the
    # next real request. Only the scalar control rewards below are authored.
    calls=load(source/'rollout_0/generation.json')
    prefix=len(calls[1]['prompt_ids'])-len(original[0].prompt_ids)
    short=original[0].model_copy(deep=True)
    short.response_ids=short.response_ids[:prefix];short.response_mask=short.response_mask[:prefix];short.response_logprobs=short.response_logprobs[:prefix];short.num_turns=3
    assert short.response_mask[-1]==0 and short.response_ids==calls[1]['prompt_ids'][len(short.prompt_ids):]
    controls=[]
    for output,reward in ((original[0],1.),(original[1],-1.),(short,1.),(short,0.)):
        item=output.model_copy(deep=True);item.reward_score=reward;item.extra_fields={'authored_reward_control':True};item.metrics=AgentLoopMetrics();controls.append(item)
    controlled=await process(controls)
    index=np.array(['full','full','tool_tail','tool_tail'],dtype=object)
    advantage,_=compute_grpo_outcome_advantage(controlled.batch['rm_scores'],controlled.batch['response_mask'],index)
    mask=controlled.batch['response_mask']
    for row,rewards,offset in ((0,[1.,-1.],0),(1,[1.,-1.],1),(2,[1.,0.],0),(3,[1.,0.],1)):
        expected=relative_advantages(rewards)[offset]
        assert torch.allclose(advantage[row][mask[row].bool()],torch.full_like(advantage[row][mask[row].bool()],expected),atol=1e-6,rtol=0)
    assert not advantage[~mask.bool()].any()
    old=controlled.batch['rollout_log_probs'];leaf=old.clone().requires_grad_(True)
    actor=ActorConfig(strategy='fsdp',rollout_n=2,ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=1)
    loss,metrics=compute_policy_loss_vanilla(old,leaf,advantage,mask,config=actor)
    loss.backward();gradient=leaf.grad
    expected_gradient=-advantage*mask/mask.sum()
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert torch.allclose(gradient,expected_gradient,atol=1e-7,rtol=0)
    assert not gradient[~mask.bool()].any() and gradient[mask.bool()].count_nonzero()==mask.sum()
    # A literal pad-ID inside the observed sequence is still a real token.
    pad=AgentLoopOutput(prompt_ids=original[0].prompt_ids,response_ids=[native.tokenizer.pad_token_id,42],response_mask=[1,0],response_logprobs=[-.5,0.],reward_score=1.,num_turns=1,metrics=AgentLoopMetrics())
    pad_result=await process([pad]);assert pad_result.batch['attention_mask'][0,8192].item()==1
    for name,batch in (('processed_real',real),('controlled_rewards_not_training',controlled)):
        write_json(root/(name+'.json'),{key:value.tolist() for key,value in batch.batch.items()})
    write_json(root/'controlled_gradient.json',dict(advantages=advantage.tolist(),logprob_leaf_gradients=gradient.tolist(),response_mask=mask.tolist(),authored_rewards=[1.,-1.,1.,0.]))
    inputs=[source/'report.json',source/'resource_plan.json']+[source/f'rollout_{i}/{name}.json' for i in range(2) for name in ('output','generation')]
    write_json(root/'report.json',dict(passed=True,kind='recorded-real-token-worker-postprocess-and-authored-reward-math-not-training',source_hashes=hashes,input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},worker_constructor_used=True,input_metadata_preserved=True,loaded_tools=len(native.tools),real_replayed_trajectories=2,real_rewards=[output.reward_score for output in original],real_effective_groups=0,authored_reward_controls=4,pad_id_control=True,tool_tail_reward_index=prefix-1,tool_tail_reward_mask=0,actor_gradient_positions=int(mask.sum()),environment_gradient_positions=int(gradient[~mask.bool()].count_nonzero()),logprob_leaf_gradient_l2=float(gradient.norm()),model_generations=0,model_weight_loads=0,optimizer_updates=0,teacher_samples=0,actor_reference_verified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--model',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:asyncio.run(worker(args.source,args.model,args.out))
    else:
        from .process_tree import execute_tree_bounded
        root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
        write_json(root/'resource_plan.json',dict(max_wall_seconds=180,max_input_file_bytes=8*1024**2,max_prompt_tokens=8192,max_response_tokens=4096,max_total_context_tokens=8192,max_output_bytes=64*1024**2,new_model_generations=0,max_api_cost_usd=0,gpu_jobs=0,advantage_atol=1e-6,gradient_atol=1e-7))
        for name in ('worker_postprocess_probe.py','groups.py','verl_loop_sources.json'):(root/('executed_'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
        with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
            status=execute_tree_bounded([sys.executable,'-m','officegym.training.worker_postprocess_probe','--worker','--source',str(Path(args.source).resolve()),'--model',str(Path(args.model).resolve()),'--out',str(root)],stdout,stderr,180,evidence_dir=root/'ownership')
        if status['exit_code'] or status['wall_budget_exhausted']:raise SystemExit(1)
        if sum(path.stat().st_size for path in root.rglob('*') if path.is_file())>64*1024**2:raise RuntimeError('postprocess artifact byte cap')
        print((root/'report.json').read_text())

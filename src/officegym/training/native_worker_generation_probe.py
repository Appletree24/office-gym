"""Live native Ray AgentLoopWorker generation, using a verified frozen base only."""
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
from uuid import uuid4
from officegym.eval.artifacts import write_json


def hydra_dataclass(value):
    from dataclasses import is_dataclass,fields
    if is_dataclass(value):
        result={field.name:hydra_dataclass(getattr(value,field.name)) for field in fields(value) if field.name!='_target_'}
        result['_target_']=type(value).__module__+'.'+type(value).__qualname__
        return result
    if isinstance(value,dict):return {key:hydra_dataclass(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [hydra_dataclass(item) for item in value]
    return value


def build_worker_config(model,root,base):
    from omegaconf import OmegaConf
    from verl.workers.config.rollout import RolloutConfig,MultiTurnConfig,AgentLoopConfig
    from .worker_plugins import write_plugins
    root=Path(root);plugins=root/'worker_plugins';plugins.mkdir();records=write_plugins(plugins)
    rollout=RolloutConfig(name='vllm',n=2,temperature=0.,calculate_log_probs=True,prompt_length=8192,response_length=4096,tensor_model_parallel_size=1,multi_turn=MultiTurnConfig(enable=True,tool_config_path=records['tools']['path'],max_assistant_turns=5,max_user_turns=30,max_parallel_calls=1,max_tool_response_length=20000,format='hermes'),agent=AgentLoopConfig(num_workers=1,default_agent_loop='officegym',agent_loop_config_path=records['agent_loop']['path']))
    config=OmegaConf.create(dict(actor_rollout_ref=dict(rollout=hydra_dataclass(rollout),model={'_target_':'verl.workers.config.HFModelConfig','path':str(model),'trust_remote_code':False}),data={},distillation=None,trainer=dict(officegym_starting_actor_hash=base)))
    write_json(root/'worker_config_raw.json',OmegaConf.to_container(config,resolve=True))
    return config


class RecordingClient:
    def __init__(self,client,root):
        self.client=client;self.root=Path(root);self.counts={};self.total=0;self.paths=set()

    async def generate(self,**kwargs):
        from .rollout_context import ACTIVE_TRAJECTORY
        from .registry import REGISTRY
        trajectory=ACTIVE_TRAJECTORY.get()
        if not trajectory or trajectory not in REGISTRY.entries:raise ValueError('worker generation lacks task-local rollout identity')
        if set(kwargs)-set('request_id prompt_ids sampling_params image_data video_data mm_processor_output audio_data mm_processor_kwargs'.split()):raise ValueError('unexpected private metadata on model API')
        self.paths.add(str(REGISTRY.entries[trajectory]['environment']._path))
        self.total+=1;self.counts[trajectory]=self.counts.get(trajectory,0)+1
        if self.total>10 or self.counts[trajectory]>5:raise ValueError('native worker generation request cap')
        number=self.counts[trajectory]
        prompt=list(kwargs['prompt_ids']);params=dict(kwargs['sampling_params'])
        result=await self.client.generate(**kwargs)
        directory=self.root/'requests'/trajectory;directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/f'{number:04d}.json',dict(request_id=kwargs['request_id'],prompt_ids=prompt,sampling_params=params,token_ids=result.token_ids,log_probs=result.log_probs,backend_versions=result.extra_fields))
        return result


class WorkerActor:
    def __init__(self,model,root,balancer,base):
        import ray
        from verl.workers.rollout.llm_server import LLMServerClient
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        root=Path(root);config=build_worker_config(model,root,base)
        self.recorder=RecordingClient(LLMServerClient(config,load_balancer_handle=balancer),root)
        self.worker=AgentLoopWorker(config,self.recorder);self.root=root
        write_json(root/'worker_identity.json',dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),actor_name=ray.get_runtime_context().get_actor_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),native_worker_class=type(self.worker).__module__+'.'+type(self.worker).__name__))

    async def generate(self,batch):
        from .registry import REGISTRY
        output=await self.worker.generate_sequences(batch)
        assert not REGISTRY.entries and all(not Path(path).exists() for path in self.recorder.paths)
        write_json(self.root/'worker_cleanup.json',dict(registry_empty=True,world_paths=sorted(self.recorder.paths),worlds_removed=True,generation_requests=self.recorder.total))
        return output


async def worker(fetch_root,instance,root):
    import numpy as np
    import ray
    from verl import DataProto
    from .fetch_model import validate_plan,check_file
    from .native_ray_backend import NativeRayBackend
    from .actor_version import bind_actor_metadata,validate_actor_batch
    from .native_groups import NativeGroupWindow
    from officegym.harness.loop import initial_messages
    root=Path(root);source=Path(fetch_root);instance=Path(instance).resolve()
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','vllm','peft','ray')}
    if versions!={'torch':'2.13.0+cu130','transformers':'5.16.1','vllm':'0.32.0','peft':'0.19.1','ray':'2.55.1'}:raise ValueError('native worker runtime differs from lock')
    write_json(root/'runtime.json',versions)
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan);accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('accepted model fetch required')
    model=source/'model'
    for item in plan['files']:
        if not check_file(model/item['name'],item):raise ValueError('model bytes changed before native worker launch')
    before=hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()
    backend=await NativeRayBackend().start(model,root)
    base='base-worker-probe-'+plan['revision'];name='officegym_loop_worker_'+uuid4().hex
    actor=ray.remote(WorkerActor).options(num_cpus=1,num_gpus=0,max_concurrency=4,name=name).remote(str(model),str(root),backend.balancer,base)
    task=json.loads((instance/'task.json').read_text());public=initial_messages(task['public_task'],full=True)
    prompt=np.empty(2,dtype=object);prompt[:]=[public,public]
    metadata=[bind_actor_metadata(dict(instance_dir=str(instance),actor_version=base,max_generation_tokens=512,max_context_tokens=8192,max_tool_calls=4,token_ledger_path=str(root/'tokens.sqlite'),global_rollout_token_limit=1024,artifact_root=str(root/'trajectories')),base,0) for _ in range(2)]
    inputs=DataProto.from_dict(non_tensors=dict(raw_prompt=prompt,officegym=np.array(metadata,dtype=object),agent_name=np.array(['officegym']*2,dtype=object),uid=np.array(['one_real_instance_pair']*2,dtype=object),index=np.array([0,0])),meta_info=dict(global_steps=0))
    output=await asyncio.wait_for(actor.generate.remote(inputs),240)
    validate_actor_batch(output.non_tensor_batch,base,0,expected_rows=2)
    write_json(root/'native_batch_tensors.json',{key:value.tolist() for key,value in output.batch.items()})
    rows=[]
    for index in range(2):
        trajectory=output.non_tensor_batch['officegym_trajectory_id'][index]
        archive=root/'trajectories'/trajectory;terminal=json.loads((archive/'terminal.json').read_text())
        files=sorted((root/'requests'/trajectory).glob('*.json'));calls=[json.loads(path.read_text()) for path in files]
        mask=output.batch['response_mask'][index];attention=output.batch['attention_mask'][index]
        pl=int(attention[:8192].sum());rl=int(attention[8192:].sum())
        prompt_ids=output.batch['prompts'][index,-pl:].tolist();ids=output.batch['responses'][index,:rl].tolist();bits=mask[:rl].tolist();probs=output.batch['rollout_log_probs'][index,:rl].tolist()
        assert ids==terminal['original_token_ids'] and bits==terminal['response_mask'] and probs==terminal['original_logprobs']
        assert [t for call in calls for t in call['token_ids']]==[t for t,m in zip(ids,bits) if m]
        assert [p for call in calls for p in call['log_probs']]==[p for p,m in zip(probs,bits) if m]
        assert all(math.isfinite(p) and (m or p==0) for p,m in zip(probs,bits))
        for call in calls:
            prefix=call['prompt_ids'];offset=len(prefix)-pl
            assert prefix==prompt_ids+ids[:offset]
            assert ids[offset:offset+len(call['token_ids'])]==call['token_ids']
        assert terminal['reward']==float(output.batch['rm_scores'][index].sum()) and terminal['tool_calls']>=1
        assert not mask[rl:].any() and not output.batch['rollout_log_probs'][index,rl:].any()
        assert output.non_tensor_batch['officegym'][index]==metadata[index]
        rows.append(dict(trajectory_id=trajectory,generation_requests=len(calls),generated_tokens=sum(bits),environment_tokens=rl-sum(bits),tool_calls=terminal['tool_calls'],reward=terminal['reward'],termination_reason=terminal['termination_reason']))
    assert len({row['trajectory_id'] for row in rows})==2
    # The target instance and greedy base are expected to remain a diagnostic,
    # regardless of whether genuine rewards happen to differ.
    selected,stats=NativeGroupWindow(base,0,1,2).add(output,output.batch['rm_scores'],{})
    write_json(root/'native_group_result.json',stats)
    with sqlite3.connect(root/'tokens.sqlite') as db:assert db.execute('SELECT sum(reserved),sum(observed) FROM reservations').fetchone()==(1024,sum(row['generated_tokens'] for row in rows))
    assert hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()==before
    write_json(root/'native_batch_metadata.json',{key:output.non_tensor_batch[key].tolist() for key in ('officegym','uid','officegym_actor_version','officegym_actor_update_step','officegym_actor_version_verified','officegym_trajectory_id','min_global_steps','max_global_steps')})
    write_json(root/'report.json',dict(passed_native_protocol=True,kind='live-native-ray-agentloopworker-base-probe-not-formal-grpo',backend='ray-worker',model_revision=plan['revision'],model_generations=sum(row['generation_requests'] for row in rows),rollouts=rows,local_http_requests=2,paid_api_requests=0,optimizer_updates=0,teacher_samples=0,worker_generate_sequences_used=True,default_agentloop_manager_used=False,physical_weight_sync_verified=False,actor_reference_verified=False,filesystem_worker_isolation_verified=False,effective_groups=stats['effective_groups'],source_snapshot_sha256=before,executed_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('executed_*')}))
    backend.shutdown()

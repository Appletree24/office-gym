"""Real pinned Qwen/FSDP2 save/restore continuity probe, without GRPO claims."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from officegym.eval.artifacts import write_json

WALL_SECONDS=600
ALLOCATOR_BYTES=48*1024**3
WATCHDOG_MIB=56*1024
ARTIFACT_BYTES=2*1024**3


def worker(fetch_root,checkout,out):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
    from torchdata.stateful_dataloader import StatefulDataLoader
    from omegaconf import OmegaConf
    from .fetch_model import validate_plan,check_file
    from .verl_checkpoint import verify_sources,extract_method,record_saved_checkpoint,SOURCE_HASHES
    from .tokenizer import SFTTokenizer
    from officegym.harness.tokenizer import PromptTokenizer
    from officegym.tools.schema import tool_schemas
    root=Path(out);source=Path(fetch_root);checkout=Path(checkout)
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','verl','torchdata','tensordict','vllm')}
    if versions['torch']!='2.13.0+cu130' or versions['transformers']!='5.16.1' or versions['peft']!='0.19.1':raise ValueError('locked modern native runtime required')
    write_json(root/'packages.json',versions);verify_sources(checkout)
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    import inspect
    if Path(inspect.getfile(FSDPCheckpointManager)).resolve()!=checkout/'verl/utils/checkpoint/fsdp_checkpoint_manager.py':raise ValueError('native manager outside pinned checkout')
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan)
    accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('verified model source required')
    def verify_weights():
        for item in plan['files']:
            if not check_file(source/'model'/item['name'],item):raise ValueError('source weights changed')
    verify_weights()
    prompt=PromptTokenizer.load(str(source/'model'),plan['revision']);mask=SFTTokenizer(prompt.tokenizer);rows=[]
    for query in ('owner','中文 Ω'):
        messages=[dict(role='system',content='Use tools; observations are evidence.'),dict(role='user',content='Read the board and search for '+query),dict(role='assistant',tool_call=dict(id='one',name='board.read',arguments={})),dict(role='tool',tool_call_id='one',content='{"ok":true,"data":{"content":{},"version":1}}'),dict(role='assistant',tool_call=dict(id='two',name='people.search',arguments=dict(query=query))),dict(role='tool',tool_call_id='two',content='{"ok":true,"data":{"items":[]}}')]
        row=mask(messages,tool_schemas());row['labels']=[t if bit else -100 for t,bit in zip(row['input_ids'],row['assistant_mask'])];rows.append(row)
    oracle=[json.loads(line) for line in Path('artifacts/real_optimizer_backend_probe_v1/protocol_samples.jsonl').read_text().splitlines()]
    write_json(root/'protocol_samples_current.json',rows)
    write_json(root/'tokenizer_alignment.json',dict(passed=rows==oracle,samples=len(rows),tokenizer=prompt.manifest(),mask=mask.mask_manifest()))
    if rows!=oracle:raise ValueError('new native tokenizer/mask differs from prior verified protocol source')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(2026);np.random.seed(2026);random.seed(2026);torch.use_deterministic_algorithms(True)
    torch.distributed.init_process_group('nccl',init_method='file://'+str(root/'dist_init'),rank=0,world_size=1)
    try:
        model,loading=AutoModelForCausalLM.from_pretrained(source/'model',dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation='sdpa',output_loading_info=True)
        write_json(root/'loading_info.json',json.loads(json.dumps(loading,default=lambda value:sorted(value) if isinstance(value,set) else repr(value))))
        if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):raise ValueError('inexact pretrained loading')
        if sum(p.numel() for p in model.parameters())!=4022468096:raise ValueError('wrong pretrained architecture')
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        model=model.to('cuda');mesh=init_device_mesh('cuda',(1,));policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
        for layer in model.get_base_model().model.layers:fully_shard(layer,mesh=mesh,mp_policy=policy)
        fully_shard(model,mesh=mesh,mp_policy=policy)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,foreach=False)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1/(step+1))
        loader=StatefulDataLoader(rows,batch_size=None,shuffle=True,generator=torch.Generator().manual_seed(314),num_workers=0);iterator=iter(loader)
        def tensor_bytes(tensor):
            value=tensor.detach()
            if hasattr(value,'to_local'):value=value.to_local()
            return value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        def fingerprint(value):
            digest=hashlib.sha256()
            def visit(item):
                if isinstance(item,torch.Tensor):digest.update(str((item.dtype,tuple(item.shape))).encode());digest.update(tensor_bytes(item))
                elif isinstance(item,np.ndarray):digest.update(str((item.dtype,item.shape)).encode());digest.update(item.tobytes())
                elif isinstance(item,dict):
                    for key in sorted(item,key=repr):digest.update(repr(key).encode());visit(item[key])
                elif isinstance(item,(list,tuple)):
                    for part in item:visit(part)
                else:digest.update(repr(item).encode())
            visit(value);return digest.hexdigest()
        def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(0))
        def adapter_state():return {n:p for n,p in model.named_parameters() if p.requires_grad}
        def step(row):
            model.train();optimizer.zero_grad(set_to_none=True)
            ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
            output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)
            with torch.no_grad():
                targets=labels[:,1:];valid=targets!=-100
                expected=torch.nn.functional.cross_entropy(output.logits[:,:-1][valid].float(),targets[valid])
                if not torch.allclose(output.loss.detach().float(),expected,atol=1e-4,rtol=1e-4):raise ValueError('native masked loss differs')
            loss=output.loss.item();output.loss.backward();norm=0.
            for p in model.parameters():
                if not p.requires_grad and p.grad is not None:raise ValueError('base received gradients')
                if p.requires_grad and p.grad is not None:
                    g=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
                    if not bool(torch.isfinite(g).all()):raise ValueError('nonfinite native gradients')
                    norm+=g.float().square().sum().item()
            if norm<=0:raise ValueError('zero native adapter gradients')
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            return dict(loss=loss,gradient_squared_norm=norm,input_sha256=hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest())
        first=step(next(iterator));step1_model=fingerprint(adapter_state());step1_opt=fingerprint(optimizer.state_dict());step1_sched=scheduler.state_dict()
        cfg=OmegaConf.create(dict(data={'train_batch_size':1},trainer=dict(total_training_steps=3,use_v1=False,save_freq=1,officegym_evidence_root=str(root),default_local_dir=str(root/'checkpoints'),officegym_verl_checkout=str(checkout),resume_mode='disable',default_hdfs_dir=None),actor_rollout_ref=dict(rollout={'n':2},actor=dict(strategy='fsdp2',ppo_epochs=1,ppo_mini_batch_size=1,checkpoint=dict(async_save=False,save_contents=['model','optimizer','extra'],load_contents=['model','optimizer','extra'],save_lora_only=True)))))
        manager=FSDPCheckpointManager(model,optimizer,lr_scheduler=scheduler,processing_class=prompt.tokenizer,checkpoint_config=cfg.actor_rollout_ref.actor.checkpoint)
        save=extract_method(checkout/'verl/trainer/ppo/ray_trainer.py',SOURCE_HASHES['verl/trainer/ppo/ray_trainer.py'],'RayPPOTrainer','_save_checkpoint',dict(os=os,torch=torch))
        class WorkerGroup:
            def save_checkpoint(self,local_path,remote,step,max_ckpt_to_keep):return manager.save_checkpoint(local_path,remote,step,max_ckpt_to_keep)
        class Controller:
            _save_checkpoint=save
        controller=Controller();controller.config=cfg;controller.global_steps=1;controller.use_critic=False;controller.actor_rollout_wg=WorkerGroup();controller.train_dataloader=loader
        controller._save_checkpoint();record_saved_checkpoint(cfg,1)
        expected_rng=fingerprint(rng_state())
        uninterrupted=step(next(iterator));expected_model=fingerprint(adapter_state());expected_opt=fingerprint(optimizer.state_dict());expected_sched=scheduler.state_dict()
        if expected_model==step1_model:raise ValueError('native second optimizer update changed no parameter')
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:p.add_(.125)
        random.seed(999);np.random.seed(999);torch.manual_seed(999);scheduler.last_epoch=999
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor):value.zero_()
        actor=root/'checkpoints/global_step_1/actor';manager.load_checkpoint(str(actor),del_local_after_load=False)
        if fingerprint(adapter_state())!=step1_model or fingerprint(optimizer.state_dict())!=step1_opt or scheduler.state_dict()!=step1_sched:raise ValueError('native model/optimizer/scheduler restoration differs')
        if fingerprint(rng_state())!=expected_rng:raise ValueError('native Python/NumPy/CPU/CUDA RNG restoration differs')
        data=torch.load(root/'checkpoints/global_step_1/data.pt',weights_only=False);loader.load_state_dict(data);iterator=iter(loader)
        resumed=step(next(iterator))
        equivalent=fingerprint(adapter_state())==expected_model and fingerprint(optimizer.state_dict())==expected_opt and scheduler.state_dict()==expected_sched and resumed==uninterrupted
        if not equivalent:raise ValueError('resumed continuation differs from uninterrupted training')
        verify_weights()
        write_json(root/'report.json',dict(kind='real_qwen_single_gpu_fsdp2_checkpoint_continuity_not_grpo',model_revision=plan['revision'],passed=True,native_manager_source=SOURCE_HASHES['verl/utils/checkpoint/fsdp_checkpoint_manager.py'],base_parameters=4022468096,optimizer_updates=3,logical_final_step=2,first=first,uninterrupted=uninterrupted,resumed=resumed,continuation_bitwise_equal=True,model_optimizer_scheduler_restored=True,dataloader_rng_restored=True,source_weights_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_samples=0,formal_sft=False,grpo_executed=False))
    finally:torch.distributed.destroy_process_group()


def run(fetch_root,checkout,out):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
    write_json(root/'plan.json',dict(kind='real_fsdp2_backend_probe',model_fetch_root=str(Path(fetch_root).resolve()),source_checkout=str(Path(checkout).resolve()),max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,max_allocated_bytes=ALLOCATOR_BYTES,gpu_watchdog_mib=WATCHDOG_MIB,max_artifact_bytes=ARTIFACT_BYTES,max_api_cost_usd=0,max_optimizer_updates=3,formal_training=False))
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if idle.stdout.strip():raise RuntimeError('GPU must be idle; existing jobs preserved')
    env=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_IB_DISABLE='1',TORCHINDUCTOR_CACHE_DIR=str(root/'torch_cache'),TRITON_CACHE_DIR=str(root/'triton_cache'),VERL_USE_EXTERNAL_PLUGINS='none',VERL_USE_EXTERNAL_MODULES='')
    started=time.monotonic();peak=0
    with (root/'worker.stdout').open('w') as stdout,(root/'worker.stderr').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.training.fsdp_probe','--worker','--fetch-root',str(Path(fetch_root).resolve()),'--checkout',str(Path(checkout).resolve()),'--out',str(root)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        write_json(root/'process.json',dict(pid=child.pid,pgid=child.pid))
        try:
            while child.poll() is None:
                if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('native FSDP probe wall cap')
                if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>ARTIFACT_BYTES:raise RuntimeError('native checkpoint disk cap')
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True);used=0
                for line in query.stdout.splitlines():
                    pid,memory=line.split(',')
                    try:
                        if os.getpgid(int(pid))==child.pid:used+=int(memory)
                    except ProcessLookupError:pass
                peak=max(peak,used)
                if used>WATCHDOG_MIB:raise RuntimeError('native FSDP GPU watchdog')
                time.sleep(1)
            if child.returncode:raise RuntimeError('native FSDP worker failed; retained logs')
            report=json.loads((root/'report.json').read_text())
            if not report.get('passed'):raise RuntimeError('native checkpoint evidence failed')
            write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak));return report
        except BaseException as exc:
            write_json(root/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak));raise
        finally:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--checkout',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.worker:worker(Path(args.fetch_root),Path(args.checkout),Path(args.out))
    else:print(json.dumps(run(args.fetch_root,args.checkout,args.out)))

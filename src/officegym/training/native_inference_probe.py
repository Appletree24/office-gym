"""Bounded real vLLM 0.32 token generation through pinned OfficeGym AgentLoop.

Uses the verified original base model, not a formally admitted SFT/GRPO actor.
"""
import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sqlite3
import tempfile
import secrets
import sys
import time
from officegym.eval.artifacts import write_json

WALL_SECONDS=600
GPU_MIB=36*1024
ARTIFACT_BYTES=2*1024**3


def descendant(pid,ancestor):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==ancestor:return True
        seen.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[1])
        except (FileNotFoundError,ProcessLookupError):return False
    return False


async def worker(fetch_root,instance,root,backend='direct'):
    from .fetch_model import validate_plan,check_file
    from .worker_plugins import plugin_contents
    from .verl_loop import OfficeGymAgentLoop
    from .verl_tool import OfficeGymTool
    from .registry import REGISTRY
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap,ToolListWrap
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.workers.rollout.replica import TokenOutput
    from vllm import LLM,SamplingParams
    source=Path(fetch_root);instance=Path(instance).resolve();root=Path(root)
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','vllm','peft','ray')}
    expected={'torch':'2.13.0+cu130','transformers':'5.16.1','vllm':'0.32.0','peft':'0.19.1','ray':'2.55.1'}
    if versions!=expected:raise ValueError('native model probe runtime differs from explicit lock')
    write_json(root/'runtime.json',versions)
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan)
    accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('accepted verified model fetch required')
    model=source/'model'
    for item in plan['files']:
        if not check_file(model/item['name'],item):raise ValueError('model bytes changed')
    before=hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    ray_backend=None
    if backend=='ray':
        from .native_ray_backend import NativeRayBackend
        ray_backend=await NativeRayBackend().start(model,root)
    else:
        engine=LLM(model=str(model),dtype='bfloat16',max_model_len=8192,max_num_seqs=1,gpu_memory_utilization=.30,enforce_eager=True,enable_prefix_caching=False,seed=17)
    with contextlib.redirect_stdout(io.StringIO()):
        tools=[OfficeGymTool(row['config'],OpenAIFunctionToolSchema.model_validate(row['tool_schema'])) for row in plugin_contents()['tools']['tools']]
    rows=[];paths=set()
    class Server:
        def __init__(self,directory):self.directory=directory;self.calls=[]
        async def generate(self,**kwargs):
            assert set(kwargs)<=set('request_id prompt_ids sampling_params image_data video_data mm_processor_output audio_data mm_processor_kwargs'.split())
            paths.update(str(entry['environment']._path) for entry in REGISTRY.entries.values())
            params=kwargs['sampling_params'];cap=params['max_tokens']
            if len(self.calls)>=5:raise ValueError('native model generation request cap')
            if ray_backend is not None:
                result=await ray_backend.generate(**kwargs)
                ids=list(result.token_ids);probs=result.log_probs
                reason=result.stop_reason;fields=result.extra_fields
            else:
                sampling=SamplingParams(max_tokens=cap,temperature=0,logprobs=0,seed=17,stop_token_ids=params.get('stop_token_ids'))
                outputs=await asyncio.to_thread(engine.generate,{'prompt_token_ids':list(kwargs['prompt_ids'])},sampling,use_tqdm=False)
                if len(outputs)!=1 or len(outputs[0].outputs)!=1:raise ValueError('single native completion required')
                result=outputs[0].outputs[0];ids=list(result.token_ids)
                probs=None if result.logprobs is None else [float(row[token].logprob) for row,token in zip(result.logprobs,ids)]
                reason=result.finish_reason;fields={}
            if not ids or len(ids)>cap or probs is None or len(probs)!=len(ids):raise ValueError('invalid original engine token/logprob result')
            if any(not math.isfinite(p) for p in probs):raise ValueError('nonfinite native generation logprob')
            self.calls.append(dict(prompt_ids=list(kwargs['prompt_ids']),token_ids=ids,log_probs=probs,finish_reason=reason,max_tokens=cap,backend_versions=fields))
            write_json(self.directory/'generation.json',self.calls)
            return TokenOutput(token_ids=ids,log_probs=probs,extra_fields=fields)
    for index in range(2):
        directory=root/f'rollout_{index}';directory.mkdir()
        config=OmegaConf.create(dict(data={},actor_rollout_ref=dict(rollout=dict(prompt_length=8192,response_length=4096,full_determinism=False,multi_turn=dict(max_user_turns=30,max_assistant_turns=5,max_parallel_calls=1,max_tool_response_length=20000,tool_response_truncate_side='right',format='hermes')))))
        write_json(directory/'loop_config.json',OmegaConf.to_container(config,resolve=True))
        server=Server(directory)
        loop=OfficeGymAgentLoop(trainer_config=DictConfigWrap(config),server_manager=server,tokenizer=tokenizer,processor=None,dataset_cls=object,data_config=DictConfigWrap(config.data),hf_model_type='qwen3',tools=ToolListWrap(tools))
        trusted=dict(actor_version='base-probe-'+plan['revision'],max_generation_tokens=512,max_context_tokens=8192,token_ledger_path=str(root/'tokens.sqlite'),global_rollout_token_limit=1024,instance_dir=str(instance),artifact_root=str(directory/'trajectories'),max_tool_calls=4)
        if ray_backend is not None:
            from .actor_version import bind_actor_metadata
            trusted=bind_actor_metadata(trusted,trusted['actor_version'],0)
        output=await loop.run(dict(max_tokens=128,temperature=0,logprobs=True),officegym=trusted)
        assert output.extra_fields['officegym_actor_version_verified']==(ray_backend is not None)
        write_json(directory/'output.json',output.model_dump())
        ids=[t for call in server.calls for t in call['token_ids']];probs=[p for call in server.calls for p in call['log_probs']]
        assert ids==[t for t,m in zip(output.response_ids,output.response_mask) if m]
        assert probs==[p for p,m in zip(output.response_logprobs,output.response_mask) if m]
        assert all(p==0 for p,m in zip(output.response_logprobs,output.response_mask) if not m)
        assert len(output.response_ids)==len(output.response_mask)==len(output.response_logprobs)
        assert len(output.prompt_ids)+len(output.response_ids)<=8192
        terminal=json.loads(next((directory/'trajectories').glob('*/terminal.json')).read_text())
        assert terminal['original_token_ids']==output.response_ids and terminal['original_logprobs']==output.response_logprobs
        assert terminal['tool_calls']>=1 and not REGISTRY.entries
        rows.append(dict(rollout=index,generation_requests=len(server.calls),generated_tokens=len(ids),environment_tokens=len(output.response_ids)-sum(output.response_mask),reward=terminal['reward'],tool_calls=terminal['tool_calls'],termination_reason=terminal['termination_reason']))
    assert len(paths)==2 and all(not Path(path).exists() for path in paths)
    with sqlite3.connect(root/'tokens.sqlite') as db:
        assert db.execute('SELECT sum(reserved),sum(observed) FROM reservations').fetchone()==(1024,sum(r['generated_tokens'] for r in rows))
    assert hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()==before
    local_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('executed_*')}
    write_json(root/'report.json',dict(passed_native_protocol=True,kind='base-model-vllm-native-loop-not-formal-grpo',backend=backend,physical_weight_sync_verified=False,local_http_requests=2 if ray_backend else 0,model_revision=plan['revision'],model_fetch_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),executed_source_hashes=local_hashes,model_generations=sum(row['generation_requests'] for row in rows),optimizer_updates=0,teacher_samples=0,api_requests=2 if ray_backend else 0,paid_api_requests=0,rollouts=rows,isolated_world_paths=sorted(paths),source_snapshot_sha256=before,filesystem_worker_isolation_verified=False,actor_reference_verified=False))

    if ray_backend is not None:ray_backend.shutdown()


def run(fetch_root,instance,out,backend='direct'):
    if backend not in ('direct','ray','ray-worker','ray-manager','native-logprob','ray-weight-sync'):raise ValueError('explicit direct/ray probe backend required')
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    artifact_dirs=[root]
    if backend.startswith('ray'):
        runtime=Path(tempfile.mkdtemp(prefix='ogray-native-'));artifact_dirs.append(runtime)
        (root/'ray_runtime_path.txt').write_text(str(runtime))
    def artifact_bytes():
        total=0
        for directory in artifact_dirs:
            for path in directory.rglob('*'):
                try:
                    if path.is_file():total+=path.stat().st_size
                except FileNotFoundError:pass
        return total
    write_json(root/'resource_plan.json',dict(backend=backend,max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,max_gpu_mib=GPU_MIB,max_artifact_bytes=ARTIFACT_BYTES,max_api_cost_usd=0,max_rollouts=2,max_generation_requests=10,max_generated_tokens=1024,fetch_root=str(Path(fetch_root).resolve()),instance=str(Path(instance).resolve())))
    for name in ('native_inference_probe.py','verl_loop.py','process_tree.py','verl_loop_sources.json','native_ray_backend.py','native_worker_generation_probe.py','rollout_context.py','native_logprob_probe.py','native_weight_sync_probe.py'):
        (root/('executed_'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
    if idle.stdout.strip():
        write_json(root/'failure.json',dict(stage='idle_preflight',error='GPU occupied; existing jobs preserved'))
        raise RuntimeError('idle GPU required')
    started=time.monotonic();peak=0;last=0
    def monitor(supervisor_pid):
        nonlocal peak,last
        if time.monotonic()-last<1:return
        last=time.monotonic()
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
        used=0
        for line in query.stdout.splitlines():
            pid,memory=line.split(',')
            if descendant(int(pid),supervisor_pid):used+=int(memory)
        peak=max(peak,used)
        if used>GPU_MIB:raise RuntimeError('native vLLM owned GPU memory watchdog')
        total=artifact_bytes()
        if total>ARTIFACT_BYTES:raise RuntimeError('native vLLM artifact byte watchdog')
    from .process_tree import execute_tree_bounded
    # Clean, explicit environment; no inherited API credentials.
    environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_CACHE_ROOT=str(root/'vllm_cache'),TORCHINDUCTOR_CACHE_DIR=str(root/'torch_cache'),RAY_USAGE_STATS_ENABLED='0')
    if backend.startswith('ray'):environment['VLLM_API_KEY']=secrets.token_urlsafe(32)
    command=[sys.executable,'-m','officegym.training.native_inference_probe','--worker','--fetch-root',str(Path(fetch_root).resolve()),'--instance',str(Path(instance).resolve()),'--out',str(root),'--backend',backend]
    try:
        with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
            status=execute_tree_bounded(command,stdout,stderr,WALL_SECONDS,evidence_dir=root/'ownership',monitor=monitor,env=environment)
        if status['exit_code'] or status['wall_budget_exhausted']:raise RuntimeError('native model worker failed; inspect retained logs')
        report=json.loads((root/'report.json').read_text())
        total=artifact_bytes()
        if total>ARTIFACT_BYTES:raise RuntimeError('native vLLM final artifact byte cap')
        write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak,artifact_bytes_before_resource_result=total,process_tree_cleanup=status['process_tree_cleanup']))
        return report
    except BaseException as error:
        write_json(root/'failure.json',dict(error_type=type(error).__name__,message=str(error),wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--instance',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);parser.add_argument('--backend',choices=('direct','ray','ray-worker','ray-manager','native-logprob','ray-weight-sync'),default='direct');args=parser.parse_args()
    if args.worker:
        if args.backend=='ray-weight-sync':
            from .native_weight_sync_probe import worker as sync_worker
            asyncio.run(sync_worker(args.fetch_root,args.instance,args.out))
        elif args.backend=='native-logprob':
            from .native_logprob_probe import worker as logprob_worker
            asyncio.run(logprob_worker(args.fetch_root,args.instance,args.out))
        elif args.backend in ('ray-worker','ray-manager'):
            from .native_worker_generation_probe import worker as live_worker
            asyncio.run(live_worker(args.fetch_root,args.instance,args.out,use_manager=args.backend=='ray-manager'))
        else:asyncio.run(worker(args.fetch_root,args.instance,args.out,args.backend))
    else:print(json.dumps(run(args.fetch_root,args.instance,args.out,args.backend)))

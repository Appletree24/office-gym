"""Actual pinned Ray/vLLM server, balancer and token client for bounded probes.

The loaded immutable base is explicitly tagged step zero. No weight update or
physical actor-to-rollout synchronization is implied by that diagnostic tag.
"""
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from uuid import uuid4
from officegym.eval.artifacts import write_json


class NativeRayBackend:
    async def start(self,model,root):
        import ray
        from omegaconf import OmegaConf
        from verl.workers.config.model import HFModelConfig
        from verl.workers.config.rollout import RolloutConfig
        from verl.workers.rollout.replica import RolloutMode
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        from verl.workers.rollout.router import GlobalRequestLoadBalancer
        from verl.workers.rollout.llm_server import LLMServerClient
        import inspect
        from .worker_plugins import verify_loop_sources
        root=Path(root);write_json(root/'native_backend_source_lock.json',verify_loop_sources(Path(inspect.getfile(vLLMHttpServer)).resolve().parents[4]))
        runtime=Path((root/'ray_runtime_path.txt').read_text())
        ray.init(address='local',num_cpus=4,num_gpus=1,include_dashboard=False,object_store_memory=80*1024**2,_memory=1024**3,_node_ip_address='127.0.0.1',_temp_dir=str(runtime),log_to_driver=False,namespace='officegym-'+uuid4().hex)
        config=RolloutConfig(name='vllm',n_gpus_per_node=1,nnodes=1,tensor_model_parallel_size=1,max_model_len=8192,prompt_length=8192,response_length=4096,max_num_seqs=1,max_num_batched_tokens=8192,gpu_memory_utilization=.30,enforce_eager=True,enable_sleep_mode=False,free_cache_engine=False,enable_prefix_caching=False,load_format='auto',calculate_log_probs=True,seed=17)
        write_json(root/'native_server_config.json',asdict(config))
        model_config=HFModelConfig(path=str(model),trust_remote_code=False,lora_rank=0)
        server_name='officegym_vllm_'+uuid4().hex
        self.server=ray.remote(vLLMHttpServer).options(num_cpus=1,num_gpus=1,max_concurrency=16,name=server_name).remote(config=config,model_config=model_config,rollout_mode=RolloutMode.STANDALONE,workers=[],replica_rank=0,node_rank=0,gpus_per_node=1,nnodes=1,cuda_visible_devices='0')
        await asyncio.wait_for(self.server.launch_server.remote(),240)
        address,port=await self.server.get_server_address.remote()
        endpoint=f'http://{address}:{port}/v1/models'
        def check_http():
            try:
                with urlopen(endpoint,timeout=10):pass
            except HTTPError as error:
                if error.code!=401:raise
            else:raise ValueError('native HTTP server did not require authentication')
            with urlopen(Request(endpoint,headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']}),timeout=10) as response:
                payload=json.loads(response.read(1024*1024))
                if not payload.get('data'):raise ValueError('native HTTP model identity missing')
            return dict(unauthenticated_status=401,authenticated_status=200,endpoint=endpoint,models=payload)
        write_json(root/'http_auth_check.json',await asyncio.to_thread(check_http))
        # This declares the known immutable, verified base version. It is not a
        # checkpoint-engine update and the report must keep sync_verified=false.
        await self.server.set_global_steps.remote(0)
        balancer_name='officegym_balancer_'+uuid4().hex
        self.balancer=ray.remote(GlobalRequestLoadBalancer).options(num_cpus=0,name=balancer_name).remote({'owned_base':self.server})
        client_config=OmegaConf.create(dict(actor_rollout_ref=dict(rollout=dict(name='vllm',full_determinism=False))))
        self.client=LLMServerClient(client_config,load_balancer_handle=self.balancer)
        write_json(root/'ray_backend.json',dict(ray_version=ray.__version__,server_actor_name=server_name,balancer_actor_name=balancer_name,base_step_tag=0,physical_weight_sync_verified=False,ray_runtime_path=str(runtime)))
        return self

    async def generate(self,**kwargs):return await self.client.generate(**kwargs)

    def shutdown(self):
        import ray
        ray.shutdown()

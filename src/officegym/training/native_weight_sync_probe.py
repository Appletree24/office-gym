"""Bounded real native bucketed LoRA transfer to the actual Ray/vLLM engine.

Uses original-base diagnostic adapters; step tags are transfer sequence numbers,
not optimizer steps. No formal training or admitted SFT export is implied.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from uuid import uuid4
from officegym.eval.artifacts import write_json


async def worker(fetch_root,adapter_root,out):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
    from .native_ray_backend import NativeRayBackend
    from .fetch_model import validate_plan,check_file
    root=Path(out);source=Path(fetch_root);adapters=Path(adapter_root)
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan)
    accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('accepted verified model fetch required')
    for item in plan['files']:
        if not check_file(source/'model'/item['name'],item):raise ValueError('fixed model bytes changed')
    report=json.loads((adapters/'report.json').read_text())
    if not report.get('passed') or report['model_revision']!=plan['revision'] or report['optimizer_updates']!=0:raise ValueError('matching actual native diagnostic adapters required')
    weights={};configs={}
    for label in ('initial','perturbed'):
        path=adapters/(label+'_adapter.safetensors')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['native_adapter_exports'][label]['sha256']:raise ValueError('adapter bytes changed')
        weights[label]=load_file(str(path));configs[label]=json.loads((adapters/(label+'_adapter_config.json')).read_text())
        if len(weights[label])!=504 or configs[label]['r']!=16 or configs[label]['lora_alpha']!=32:raise ValueError('unexpected native adapter layout')
    if configs['initial']!=configs['perturbed']:raise ValueError('adapter configuration mismatch')
    checkout=Path('artifacts/verl_checkout_v1/repo').resolve()
    revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if revision!='8df5baeed0f5ecdd7f870d83bc3d79194816e2a9':raise ValueError('pinned native source required')
    hashes={}
    for name in ('verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/workers/rollout/vllm_rollout/utils.py','verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py'):
        content=(checkout/name).read_bytes()
        if content!=subprocess.check_output(['git','-C',str(checkout),'show',revision+':'+name]):raise ValueError('native transport source changed')
        hashes[name]=hashlib.sha256(content).hexdigest()
    write_json(root/'transport_sources.json',dict(revision=revision,sha256=hashes))
    backend=await NativeRayBackend().start(source/'model',root,lora_rank=16)
    try:
        os.environ['RANK']='0';os.environ['RAY_LOCAL_WORLD_SIZE']='1'
        transfer_config=replace(backend.config,checkpoint_engine=replace(backend.config.checkpoint_engine,update_weights_bucket_megabytes=4))
        write_json(root/'native_transfer_config.json',asdict(transfer_config))
        sender=ServerAdapter(transfer_config,backend.model_config,device_mesh=None,replica_rank=0)
        sender.server_handle=backend.server
        tokenizer=AutoTokenizer.from_pretrained(source/'model',local_files_only=True)
        prompt=tokenizer.apply_chat_template([dict(role='user',content='Calculate 17 * 23. Reply with the result.')],tokenize=True,add_generation_prompt=True,return_dict=False)
        records=[]
        async def generate(label,version):
            output=await backend.generate(request_id=uuid4().hex,prompt_ids=prompt,sampling_params=dict(max_tokens=32,temperature=0.,logprobs=1,seed=17))
            ids=list(output.token_ids);probs=list(output.log_probs or [])
            if not ids or len(ids)>32 or len(ids)!=len(probs) or not all(math.isfinite(x) for x in probs):raise ValueError('invalid genuine rollout output')
            fields=output.extra_fields
            if any(fields.get(key)!=version for key in ('global_steps','min_global_steps','max_global_steps')):raise ValueError('rollout did not report completed transfer version')
            record=dict(label=label,token_ids=ids,log_probs=probs,version={key:fields[key] for key in ('global_steps','min_global_steps','max_global_steps')})
            records.append(record);write_json(root/'generations.json',dict(prompt_ids=prompt,records=records));return record
        base=await generate('base_before_adapter',0)
        for label,version in (('initial',0),('perturbed',1),('initial',2)):
            await sender.update_weights(iter(weights[label].items()),global_steps=version,peft_config=configs[label],base_sync_done=True)
            await generate(label,version)
        initial,changed,restored=records[1:]
        if initial['token_ids']!=restored['token_ids'] or initial['log_probs']!=restored['log_probs']:raise ValueError('native adapter restoration changed rollout')
        same_tokens=initial['token_ids']==changed['token_ids']
        difference=max((abs(a-b) for a,b in zip(initial['log_probs'],changed['log_probs'])),default=0.)
        if same_tokens and difference<=.01:raise ValueError('physical adapter perturbation not detected in rollout')
        write_json(root/'report.json',dict(passed=True,kind='native-bucketed-diagnostic-lora-rollout-sync-not-formal-training',model_revision=plan['revision'],physical_rollout_weight_sync_verified=True,transport='native ServerAdapter.update_weights / BucketedWeightSender / native vLLM worker receiver',use_shm=sender.use_shm,bucket_megabytes=4,adapter_tensors=504,adapter_sources=report['native_adapter_exports'],transfer_sequence_versions=[0,1,2],optimizer_updates=0,reference_is_admitted_sft=False,teacher_samples=0,model_generations=4,generated_tokens=sum(len(r['token_ids']) for r in records),initial_restored_bitwise_equal=True,perturbed_token_sequence_changed=not same_tokens,paired_generated_logprob_difference_max=difference,base_initial_token_sequence_equal=base['token_ids']==initial['token_ids']))
    finally:backend.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--adapter-root',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    from .native_inference_probe import run
    print(json.dumps(run(args.fetch_root,args.adapter_root,args.out,backend='ray-weight-sync')))

"""Bounded local LoRA merge; equivalence verification is a separate gate."""
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from officegym.eval.artifacts import write_json
from .checkpoints import checkpoint_fingerprint,bind_training_source
from .launch import execute_bounded


def validate_merge_plan(plan):
    required={'base_model_path','base_revision','base_checkpoint_sha256','adapter_path','training_events','optimizer_step','locked_packages','device','dtype','max_wall_seconds','max_shard_size_bytes','max_export_bytes'}
    if required-set(plan):
        raise ValueError('missing resolved merge fields: '+','.join(sorted(required-set(plan))))
    if plan['device'] not in ('cpu','cuda') or plan['dtype'] not in ('float32','float16','bfloat16'):
        raise ValueError('explicit supported merge device/dtype required')
    if plan.get('merge_accumulation','native') not in ('native','float64'):raise ValueError('unsupported merge accumulation dtype')
    if plan.get('merge_accumulation','native')=='float64' and plan['dtype']!='float32':raise ValueError('FP64 accumulation currently requires FP32 output')
    if plan['device']=='cuda' and (type(plan.get('max_allocated_bytes')) is not int or plan['max_allocated_bytes']<1):
        raise ValueError('finite CUDA allocator byte cap required')
    if type(plan['max_wall_seconds']) not in (int,float) or not math.isfinite(plan['max_wall_seconds']) or plan['max_wall_seconds']<=0:
        raise ValueError('finite merge wall budget required')
    for name in ('max_shard_size_bytes','max_export_bytes'):
        if type(plan[name]) is not int or plan[name]<1:
            raise ValueError('finite positive merge byte budget required')
    if plan['max_shard_size_bytes']>plan['max_export_bytes']:
        raise ValueError('merge shard cap exceeds export byte cap')
    for name in ('base_model_path','adapter_path','training_events'):
        if not Path(plan[name]).is_absolute():
            raise ValueError('absolute local merge source path required')
    if not plan['base_revision'] or plan['base_revision'].startswith('SET_'):
        raise ValueError('locked base revision required')
    if not isinstance(plan['locked_packages'],dict) or any(not isinstance(plan['locked_packages'].get(name),str) or not plan['locked_packages'][name] for name in ('torch','transformers','peft')):
        raise ValueError('complete merge dependency lock required')
    base=checkpoint_fingerprint(plan['base_model_path'])
    if base['checkpoint_sha256']!=plan['base_checkpoint_sha256']:
        raise ValueError('base model content differs from merge plan')
    if json.loads((Path(plan['base_model_path'])/'config.json').read_text()).get('quantization_config'):
        raise ValueError('quantized merge requires a separately verified export path')
    if not (Path(plan['adapter_path'])/'tokenizer_config.json').is_file():
        raise ValueError('saved local adapter tokenizer required')
    config=json.loads((Path(plan['adapter_path'])/'adapter_config.json').read_text())
    if config.get('peft_type')!='LORA':
        raise ValueError('this export path requires a LoRA adapter')
    source=bind_training_source(plan['training_events'],plan['adapter_path'],plan['optimizer_step'])
    return dict(base_checkpoint_sha256=base['checkpoint_sha256'],training_source=source)


def check_packages(plan):
    for package in ('torch','transformers','peft'):
        if importlib.metadata.version(package)!=plan['locked_packages'][package]:
            raise ValueError('merge runtime package differs from lock: '+package)


def configure_memory(plan):
    import torch
    if plan['device']=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA merge/equivalence requires actual GPU')
        total=torch.cuda.get_device_properties(0).total_memory
        if plan['max_allocated_bytes']>total:raise ValueError('allocator cap exceeds device memory')
        torch.cuda.set_per_process_memory_fraction(plan['max_allocated_bytes']/total)
        torch.cuda.reset_peak_memory_stats()



def merge_with_float64_accumulation(model):
    """Merge supported vanilla linear LoRA with one final cast to base dtype."""
    import torch
    from peft.tuners.lora.layer import Linear
    if list(model.peft_config)!=['default']:raise ValueError('FP64 merge requires exactly the default adapter')
    config=model.peft_config['default']
    if any(getattr(config,key,None) for key in ('use_dora','use_rslora','lora_bias','fan_in_fan_out','rank_pattern','alpha_pattern','modules_to_save','target_parameters')) or config.bias!='none':raise ValueError('FP64 merge supports vanilla linear LoRA without bias/pattern variants')
    layers=[module for module in model.modules() if isinstance(module,Linear)]
    parameters=[name for name,_ in model.named_parameters() if '.lora_' in name]
    if not layers or len(parameters)!=2*len(layers):raise ValueError('unsupported or unaccounted LoRA parameters')
    seen=set()
    for layer in layers:
        weight=layer.get_base_layer().weight
        a=layer.lora_A['default'].weight;b=layer.lora_B['default'].weight
        if layer.merged or list(layer.active_adapters)!=['default'] or layer.fan_in_fan_out or len(layer.lora_variant):raise ValueError('unsupported LoRA layer state')
        if weight.data_ptr() in seen:raise ValueError('shared adapted weights require separate verification')
        seen.add(weight.data_ptr())
        if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in (weight,a,b)):raise ValueError('finite FP32 base and adapter weights required')
    with torch.no_grad():
        for layer in layers:
            weight=layer.get_base_layer().weight
            delta=(layer.lora_B['default'].weight.double()@layer.lora_A['default'].weight.double())*layer.scaling['default']
            value=weight.double()+delta
            if not torch.isfinite(value).all():raise ValueError('nonfinite FP64 merge arithmetic')
            rounded=value.to(weight.dtype)
            if not torch.isfinite(rounded).all():raise ValueError('nonfinite FP32 merged weight')
            weight.copy_(rounded)
    return model.unload(),dict(accumulation_dtype='float64',output_dtype='float32',linear_layers=len(layers),single_final_cast=True)


def worker(plan_path,sha256,out):
    payload=Path(plan_path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=sha256:
        raise ValueError('merge plan changed before execution')
    plan=json.loads(payload)
    source=validate_merge_plan(plan)
    check_packages(plan)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    configure_memory(plan)
    model=AutoModelForCausalLM.from_pretrained(plan['base_model_path'],revision=plan['base_revision'],local_files_only=True,trust_remote_code=False,dtype=getattr(torch,plan['dtype']),attn_implementation='sdpa')
    model.to(plan['device'])
    model=PeftModel.from_pretrained(model,plan['adapter_path'],local_files_only=True,is_trainable=False)
    model.eval()
    arithmetic=dict(accumulation_dtype='native',output_dtype=plan['dtype'])
    if plan.get('merge_accumulation','native')=='float64':
        merged,arithmetic=merge_with_float64_accumulation(model)
    else:merged=model.merge_and_unload(safe_merge=True)
    tensor_bytes=sum(tensor.numel()*tensor.element_size() for tensor in merged.state_dict().values())
    if tensor_bytes>plan['max_export_bytes']:
        raise ValueError('merged tensors exceed export byte budget')
    root=Path(out)
    target=root/'merged'
    if target.exists():
        raise FileExistsError(target)
    merged.save_pretrained(target,safe_serialization=True,max_shard_size=plan['max_shard_size_bytes'])
    tokenizer=AutoTokenizer.from_pretrained(plan['adapter_path'],local_files_only=True,trust_remote_code=False)
    tokenizer.save_pretrained(target)
    actual_bytes=sum(path.stat().st_size for path in target.rglob('*') if path.is_file())
    if actual_bytes>plan['max_export_bytes']:
        raise ValueError('saved model exceeds export byte budget; partial artifacts retained')
    report=dict(source,merge_arithmetic=arithmetic,peak_allocated_bytes=torch.cuda.max_memory_allocated() if plan['device']=='cuda' else None,exported_checkpoint_sha256=checkpoint_fingerprint(target)['checkpoint_sha256'],model_path=str(target.resolve()),export_bytes=actual_bytes,merged=True,passed=False,
                equivalence={'logits':'pending','generation':'pending','tool_behavior':'pending'})
    write_json(root/'merge_result.json',report)
    return report


def merge(plan_path,out,*,own_process_group=True):
    plan=json.loads(Path(plan_path).read_text())
    source=validate_merge_plan(plan)
    check_packages(plan)
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    frozen=root/'resolved_merge_plan.json'
    write_json(frozen,plan)
    write_json(root/'merge_source.json',source)
    command=[sys.executable,'-m','officegym.training.merge','--worker-plan',str(frozen),'--sha256',hashlib.sha256(frozen.read_bytes()).hexdigest(),'--out',str(root)]
    with (root/'merge.stdout.log').open('w') as stdout,(root/'merge.stderr.log').open('w') as stderr:
        status=execute_bounded(command,stdout,stderr,plan['max_wall_seconds'],own_process_group=own_process_group)
    status['merge_completed']=status['exit_code']==0 and not status['wall_budget_exhausted'] and (root/'merge_result.json').is_file()
    status['equivalence_verified']=False
    write_json(root/'merge_process.json',status)
    return status


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan');source.add_argument('--worker-plan')
    parser.add_argument('--sha256');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=worker(args.worker_plan,args.sha256,args.out) if args.worker_plan else merge(args.plan,args.out)
    print(json.dumps(result,indent=2))
    if result.get('exit_code',0)!=0 or result.get('wall_budget_exhausted'):
        raise SystemExit(1)

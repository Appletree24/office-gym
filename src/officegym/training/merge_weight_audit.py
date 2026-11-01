"""CPU audit of saved FP32 vanilla-LoRA merges; not a logit/export admission gate."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from officegym.eval.artifacts import write_json


def worker(plan_path,result_path,out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .merge import validate_merge_plan,check_packages
    from .checkpoints import checkpoint_fingerprint
    root=Path(out);plan=json.loads(Path(plan_path).read_text());source=validate_merge_plan(plan);check_packages(plan)
    if plan['dtype']!='float32':raise ValueError('weight roundoff audit requires FP32 merge')
    merged=json.loads(Path(result_path).read_text());target=Path(merged['model_path'])
    if merged.get('training_source')!=source['training_source'] or merged.get('base_checkpoint_sha256')!=source['base_checkpoint_sha256'] or checkpoint_fingerprint(target)['checkpoint_sha256']!=merged['exported_checkpoint_sha256']:raise ValueError('merge source or saved weights changed')
    config=json.loads((Path(plan['adapter_path'])/'adapter_config.json').read_text())
    if any(config.get(key) for key in ('use_dora','use_rslora','lora_bias','fan_in_fan_out','rank_pattern','alpha_pattern','modules_to_save','target_parameters')) or config.get('bias','none')!='none':raise ValueError('only untransposed vanilla constant-rank LoRA is supported')
    rank=config['r'];alpha=config['lora_alpha'];scale=alpha/rank
    if type(rank) is not int or rank<1 or not math.isfinite(scale) or scale<=0 or math.frexp(scale)[0]!=.5:raise ValueError('positive rank and exact power-of-two LoRA scaling required')
    torch.set_num_threads(2)
    def index(directory):
        directory=Path(directory);path=directory/'model.safetensors.index.json'
        if path.exists():return json.loads(path.read_text())['weight_map']
        with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as stream:return {name:'model.safetensors' for name in stream.keys()}
    base=Path(plan['base_model_path']);original=index(base);saved=index(target)
    if original.keys()!=saved.keys():raise ValueError('merged tensor inventory differs from base')
    adapters=load_file(str(Path(plan['adapter_path'])/'adapter_model.safetensors'))
    pairs={}
    for name,a in adapters.items():
        if name.endswith('.lora_A.weight'):
            bname=name.replace('.lora_A.weight','.lora_B.weight');key=name.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
            if bname not in adapters or key not in saved:raise ValueError('adapter/base tensor mapping mismatch')
            pairs[key]=(a,adapters[bname])
    if len(pairs)*2!=len(adapters):raise ValueError('unaccounted adapter tensors')
    def shape(directory,mapping,key):
        with safe_open(str(directory/mapping[key]),framework='pt',device='cpu') as stream:return stream.get_slice(key).get_shape()
    def rows(directory,mapping,key,start,end):
        with safe_open(str(directory/mapping[key]),framework='pt',device='cpu') as stream:return stream.get_slice(key)[start:end].clone()
    u=2.**-24;gamma=rank*u/(1-rank*u);gamma64=rank*2.**-53/(1-rank*2.**-53)
    results=[];elements=0;violations=0;untouched_changes=0;rounded_differences=0
    for key in sorted(saved):
        dimensions=shape(base,original,key)
        if dimensions!=shape(target,saved,key):raise ValueError('base/merged tensor shapes differ')
        pair=pairs.get(key);maximum=0.;bad=0;different=0;changed=0;count=0
        if pair is not None:
            a,b=pair
            if a.dtype!=torch.float32 or b.dtype!=torch.float32 or list(a.shape)!=[rank,dimensions[1]] or list(b.shape)!=[dimensions[0],rank]:raise ValueError('unexpected adapter dtype/shape')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite adapter weights')
            a=a.double();b=b.double();absolute_a=a.abs()
        for start in range(0,dimensions[0],256):
            left=rows(base,original,key,start,start+256);right=rows(target,saved,key,start,start+256)
            if right.dtype!=torch.float32 or not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('nonfinite or non-FP32 saved merge')
            count+=right.numel()
            if pair is None:
                changed+=int((left.float()!=right).sum());continue
            with torch.no_grad():
                delta=(b[start:start+256]@a)*scale
                magnitude=(b[start:start+256].abs()@absolute_a)*scale
                reference=left.double()+delta
                # Normal FP32 dot-product/scaling/addition roundoff, plus FP64 reference error.
                envelope=(gamma+gamma64)*magnitude+u*delta.abs()+u*(left.double().abs()+delta.abs())
                if not torch.isfinite(reference).all() or not torch.isfinite(envelope).all():raise ValueError('nonfinite weight-audit arithmetic')
                error=(right.double()-reference).abs()
                maximum=max(maximum,float(error.max()));bad+=int((error>envelope).sum())
                different+=int((right!=reference.float()).sum())
        elements+=count;violations+=bad;untouched_changes+=changed;rounded_differences+=different
        results.append(dict(tensor=key,shape=dimensions,adapted=pair is not None,elements=count,max_absolute_error_from_fp64_merge=maximum,fp32_roundoff_envelope_violations=bad,different_from_single_rounded_fp64_merge=different,untouched_changed_elements=changed))
        write_json(root/'progress.json',dict(tensors_checked=len(results),last_tensor=key))
    # Revalidate inputs after all matrix reads; never write source model weights.
    if validate_merge_plan(plan)!=source or checkpoint_fingerprint(target)['checkpoint_sha256']!=merged['exported_checkpoint_sha256']:raise ValueError('source changed during weight audit')
    write_json(root/'report.json',dict(passed_weight_audit=violations==0 and untouched_changes==0,kind='fp32-weight-roundoff-diagnostic-not-numerical-or-formal-export-acceptance',normal_fp32_roundoff_envelope_only=True,rank=rank,scale=scale,tensors=len(results),adapted_tensors=len(pairs),elements=elements,roundoff_envelope_violations=violations,untouched_changed_elements=untouched_changes,different_from_single_rounded_fp64_merge=rounded_differences,max_absolute_error_from_fp64_merge=max(item['max_absolute_error_from_fp64_merge'] for item in results),tensor_results=results,source=source,exported_checkpoint_sha256=merged['exported_checkpoint_sha256'],optimizer_updates=0,model_generations=0,numerical_equivalence_passed=False,formal_export=False))


def run(plan,result,out):
    from .process_tree import execute_tree_bounded
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    for name,path in [('merge_plan',plan),('merge_result',result)]:
        payload=Path(path).read_bytes();(root/(name+'.json')).write_bytes(payload)
    (root/'executed_merge_weight_audit.py').write_bytes(Path(__file__).read_bytes())
    limits=dict(max_wall_seconds=300,max_worker_rss_bytes=4*1024**3,max_artifact_bytes=10*1024**2,device='cpu',model_generations=0,optimizer_updates=0,max_api_cost_usd=0)
    write_json(root/'resource_plan.json',limits);started=time.monotonic();peak=0
    def monitor(supervisor):
        nonlocal peak
        state=root/'ownership/worker.json'
        if state.exists():
            pid=json.loads(state.read_text())['pid']
            try:
                rss=next(int(line.split()[1])*1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
                peak=max(peak,rss)
                if rss>limits['max_worker_rss_bytes']:raise RuntimeError('weight audit RSS cap')
            except (FileNotFoundError,StopIteration):pass
        if sum(path.stat().st_size for path in root.rglob('*') if path.is_file())>limits['max_artifact_bytes']:raise RuntimeError('weight audit artifact cap')
    command=[sys.executable,'-m','officegym.training.merge_weight_audit','--worker','--plan',str(root/'merge_plan.json'),'--result',str(root/'merge_result.json'),'--out',str(root)]
    env=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1')
    with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
        status=execute_tree_bounded(command,stdout,stderr,300,evidence_dir=root/'ownership',monitor=monitor,env=env)
    write_json(root/'resource_result.json',dict(**status,wall_seconds_total=time.monotonic()-started,peak_worker_rss_bytes=peak))
    if status['exit_code'] or status['wall_budget_exhausted']:raise RuntimeError('weight audit worker failed; inspect retained logs')
    return json.loads((root/'report.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--result',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:
        write_json(Path(args.out)/'ownership/worker.json',dict(pid=os.getpid()));worker(args.plan,args.result,args.out)
    else:
        value=run(args.plan,args.result,args.out);print(json.dumps({key:value[key] for key in ('passed_weight_audit','tensors','roundoff_envelope_violations','untouched_changed_elements','different_from_single_rounded_fp64_merge','max_absolute_error_from_fp64_merge')}))

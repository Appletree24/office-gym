"""Bounded GPU supervision of an explicit FP64-accumulation merge candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from officegym.eval.artifacts import write_json
from .native_inference_probe import descendant
from .process_tree import execute_tree_bounded


def run(plan_path,out):
    from . import merge
    plan=json.loads(Path(plan_path).read_text());merge.validate_merge_plan(plan);merge.check_packages(plan)
    if plan.get('merge_accumulation')!='float64' or plan['dtype']!='float32' or plan['device']!='cuda':raise ValueError('explicit CUDA FP64 accumulation/FP32 output candidate required')
    if plan['max_wall_seconds']>300 or plan['max_export_bytes']>20*1024**3 or plan['max_allocated_bytes']>24*1024**3:raise ValueError('precise-merge probe exceeds bounded local profile')
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(root).free<plan['max_export_bytes']+1024**3:raise ValueError('export cap plus disk reserve required')
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if idle.stdout.strip():raise RuntimeError('idle GPU required; existing jobs preserved')
    frozen=root/'merge_plan.json';write_json(frozen,plan)
    for module in (merge,sys.modules[__name__]):
        (root/('executed_'+Path(module.__file__).name)).write_bytes(Path(module.__file__).read_bytes())
    limits=dict(max_wall_seconds=plan['max_wall_seconds'],max_observed_gpu_mib=32*1024,max_artifact_bytes=plan['max_export_bytes']+10*1024**2,max_allocated_bytes=plan['max_allocated_bytes'],max_api_cost_usd=0,optimizer_updates=0,model_generations=0,formal_export=False)
    write_json(root/'resource_plan.json',limits);started=time.monotonic();peak=0;last=0
    def monitor(supervisor):
        nonlocal peak,last
        if time.monotonic()-last<1:return
        last=time.monotonic();used=0
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        for line in result.stdout.splitlines():
            pid,memory=line.split(',')
            if descendant(int(pid),supervisor):used+=int(memory)
        peak=max(peak,used)
        if used>limits['max_observed_gpu_mib']:raise RuntimeError('owned merge GPU memory cap')
        total=0
        for path in root.rglob('*'):
            try:
                if path.is_file():total+=path.stat().st_size
            except FileNotFoundError:pass
        if total>limits['max_artifact_bytes']:raise RuntimeError('merge artifact byte cap')
    command=[sys.executable,'-m','officegym.training.merge','--worker-plan',str(frozen),'--sha256',hashlib.sha256(frozen.read_bytes()).hexdigest(),'--out',str(root)]
    env=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
        status=execute_tree_bounded(command,stdout,stderr,plan['max_wall_seconds'],evidence_dir=root/'ownership',monitor=monitor,env=env)
    write_json(root/'resource_result.json',dict(**status,wall_seconds_total=time.monotonic()-started,observed_peak_gpu_mib=peak))
    if status['exit_code'] or status['wall_budget_exhausted']:raise RuntimeError('precise merge worker failed; inspect retained logs')
    return json.loads((root/'merge_result.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    value=run(args.plan,args.out);print(json.dumps(dict(merged=value['merged'],arithmetic=value['merge_arithmetic'],exported_checkpoint_sha256=value['exported_checkpoint_sha256'],formal_export=False,numerical_equivalence='pending')))

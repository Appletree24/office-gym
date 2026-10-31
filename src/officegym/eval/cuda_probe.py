"""Bounded CUDA tensor forward/backward probe, without model weights or updates."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .artifacts import write_json

ALLOCATED_BYTE_CAP=256*1024*1024
EXPECTED_TORCH='2.8.0+cu128'


def worker(out):
    import torch
    if torch.__version__!=EXPECTED_TORCH or not torch.cuda.is_available():
        raise RuntimeError('expected locked CUDA torch runtime unavailable')
    active_started=time.monotonic()
    torch.cuda.set_device(0)
    properties=torch.cuda.get_device_properties(0)
    allocator_fraction=min(.01,ALLOCATED_BYTE_CAP/properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction,device=0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2026)
    base_a=torch.randn(128,128)
    base_b=torch.randn(128,128)
    expected=base_a.double()@base_b.double()
    scale=max(1.,expected.abs().max().item())
    rows=[]
    for name,dtype,tolerance in (('float32',torch.float32,1e-5),('bfloat16',torch.bfloat16,.02)):
        a=base_a.to(device='cuda',dtype=dtype).requires_grad_(True)
        b=base_b.to(device='cuda',dtype=dtype).requires_grad_(True)
        actual=a@b
        loss=actual.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        error=(actual.detach().cpu().double()-expected).abs().max().item()/scale
        gradients_finite=bool(torch.isfinite(a.grad).all().item() and torch.isfinite(b.grad).all().item())
        gradients_nonzero=bool(a.grad.abs().sum().item()>0 and b.grad.abs().sum().item()>0)
        rows.append(dict(dtype=name,normalized_max_abs_error=error,tolerance=tolerance,gradients_finite=gradients_finite,gradients_nonzero=gradients_nonzero,passed=math.isfinite(error) and error<=tolerance and gradients_finite and gradients_nonzero))
        del a,b,actual,loss
    peak=torch.cuda.max_memory_allocated()
    report=dict(kind='cuda_tensor_compatibility_not_model_validation',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,device=properties.name,compute_capability=list(torch.cuda.get_device_capability(0)),bfloat16_supported=torch.cuda.is_bf16_supported(),device_total_bytes=properties.total_memory,peak_allocated_bytes=peak,allocated_byte_cap=ALLOCATED_BYTE_CAP,allocator_fraction=allocator_fraction,cases=rows,cuda_active_wall_seconds=time.monotonic()-active_started,passed_tensor_probe=all(row['passed'] for row in rows) and peak<=ALLOCATED_BYTE_CAP,model_downloads=0,model_inference_calls=0,optimizer_updates=0,model_compatibility_pending=True)
    write_json(out,report)
    if not report['passed_tensor_probe']:
        raise RuntimeError('CUDA tensor checks failed')


def run(out,*,max_wall_seconds=60):
    if type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or not 0<max_wall_seconds<=60:
        raise ValueError('finite CUDA probe wall cap within 60 seconds required')
    root=Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    plan=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,kind='cuda_tensor_probe_plan_no_model_job',expected_torch=EXPECTED_TORCH,max_wall_seconds=max_wall_seconds,max_allocated_bytes=ALLOCATED_BYTE_CAP,max_device_allocator_fraction=.01,visible_device='0',shape=[128,128],optimizer_updates=0)
    write_json(root/'plan.json',plan)
    started=time.monotonic()
    try:
        occupancy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=min(10,max_wall_seconds),check=True)
        if occupancy.stdout.strip():
            raise RuntimeError('CUDA probe requires idle GPU; existing processes are preserved')
        hardware=subprocess.run(['nvidia-smi','--query-gpu=driver_version,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True,timeout=min(10,max_wall_seconds),check=True)
        write_json(root/'hardware.json',dict(driver_and_memory_csv=hardware.stdout.strip()))
        environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        remaining=max_wall_seconds-(time.monotonic()-started)
        if remaining<=0:
            raise TimeoutError('CUDA preflight exhausted wall cap')
        process=subprocess.run([sys.executable,'-m','officegym.eval.cuda_probe','--worker','--out',str(root/'raw.json')],env=environment,capture_output=True,text=True,timeout=remaining)
        (root/'worker.stdout').write_text(process.stdout)
        (root/'worker.stderr').write_text(process.stderr)
        if process.returncode:
            raise RuntimeError('CUDA worker failed with exit '+str(process.returncode))
        raw=json.loads((root/'raw.json').read_text())
        elapsed=time.monotonic()-started
        report=dict(raw,worker_returncode=process.returncode,wall_seconds=elapsed,max_wall_seconds=max_wall_seconds)
        if not raw['passed_tensor_probe'] or elapsed>max_wall_seconds:
            raise RuntimeError('CUDA probe failed acceptance or wall cap')
        write_json(root/'report.json',report)
        return report
    except BaseException as exc:
        if isinstance(exc,subprocess.TimeoutExpired):
            (root/'worker.stdout').write_bytes(exc.stdout or b'')
            (root/'worker.stderr').write_bytes(exc.stderr or b'')
        write_json(root/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started,passed_tensor_probe=False,model_compatibility_pending=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=60)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(Path(args.out))
    else:
        print(json.dumps(run(args.out,max_wall_seconds=args.max_wall_seconds)))

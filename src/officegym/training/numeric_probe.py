"""Owned GPU supervision for an already frozen numerical-equivalence plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from officegym.eval.artifacts import write_json

WALL_SECONDS=360
WATCHDOG_MIB=32*1024


def worker(plan,out):
    from .equivalence import verify
    root=Path(out)
    status=verify(plan,root/'numeric',own_process_group=False)
    result=root/'numeric/numeric_equivalence.json'
    if status['exit_code'] not in (0,1) or status['wall_budget_exhausted'] or not result.exists():raise RuntimeError('numerical worker failed; inspect logs')
    report=json.loads(result.read_text())
    write_json(root/'summary.json',dict(kind='prospective_numeric_backend_probe_not_formal_export',passed_numeric=report['passed_numeric'],full_equivalence_passed=False,comparisons=report['comparisons'],negative_control_detected=report.get('negative_control',{}).get('detected'),generation_match_rate=report['generation_match_rate']))


def run(plan,out):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    from . import equivalence,merge,launch
    for module in (sys.modules[__name__],equivalence,merge,launch):
        path=Path(module.__file__);(root/('executed_'+path.name)).write_bytes(path.read_bytes())
    frozen=root/'input_plan.json';frozen.write_bytes(Path(plan).read_bytes())
    write_json(root/'resource_plan.json',dict(max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,gpu_watchdog_mib=WATCHDOG_MIB,max_api_cost_usd=0,plan_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if idle.stdout.strip():
        write_json(root/'failure.json',dict(stage='idle_preflight',resource_accepted=False,error='GPU occupied; existing jobs preserved'))
        raise RuntimeError('GPU must be idle; existing jobs preserved')
    env=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started=time.monotonic();peak=0
    with (root/'worker.stdout').open('w') as stdout,(root/'worker.stderr').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.training.numeric_probe','--worker','--plan',str(frozen),'--out',str(root)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        write_json(root/'process.json',dict(pid=child.pid,pgid=child.pid))
        try:
            while child.poll() is None:
                if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('numerical supervisor wall cap')
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True);used=0
                for line in query.stdout.splitlines():
                    pid,memory=line.split(',')
                    try:
                        if os.getpgid(int(pid))==child.pid:used+=int(memory)
                    except ProcessLookupError:pass
                peak=max(peak,used)
                if used>WATCHDOG_MIB:raise RuntimeError('owned numerical GPU memory cap')
                time.sleep(1)
            if child.returncode:raise RuntimeError('numerical supervisor worker failed')
            report=json.loads((root/'summary.json').read_text())
            write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak))
            return report
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.worker:worker(args.plan,args.out)
    else:
        result=run(args.plan,args.out);print(json.dumps(result));raise SystemExit(0 if result['passed_numeric'] else 1)

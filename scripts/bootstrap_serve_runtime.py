#!/usr/bin/env python3
"""Bounded, isolated PyTorch CUDA dependency bootstrap; no model/GPU execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import venv


def size(path):
    total=0
    for base,dirs,files in os.walk(path):
        for name in files:
            try:
                total+=(Path(base)/name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def run(target,out):
    target,out=Path(target).resolve(),Path(out).resolve()
    if target.exists() or out.exists():
        raise ValueError('fresh environment and artifact paths required')
    lock=Path(__file__).resolve().parents[1]/'requirements-serve-runtime.lock'
    if not lock.is_file():
        raise ValueError('tested serve runtime lock is missing')
    out.mkdir(parents=True)
    wheels=out/'wheels';wheels.mkdir()
    temporary=out/'tmp';temporary.mkdir()
    caps=dict(wall_seconds=600,wheel_bytes=6*1024**3,environment_bytes=12*1024**3,temporary_bytes=6*1024**3)
    commands=[['-m','pip','download','--require-hashes','-r',str(lock),'--index-url','https://download.pytorch.org/whl/cu128','--only-binary=:all:','--dest',str(wheels),'--no-cache-dir','--timeout','15','--retries','0'],['-m','pip','install','--no-index','--find-links',str(wheels),'--report',str(out/'install_report.json'),'--require-hashes','-r',str(lock)]]
    plan=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),requirements_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),kind='dependency_bootstrap_not_model_job',torch='2.8.0+cu128',python=sys.version,target=str(target),caps=caps,commands=commands,model_downloads=0,gpu_jobs=0)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    started=time.monotonic()
    process=None
    stage='create_environment'
    try:
        venv.EnvBuilder(with_pip=True).create(target)
        python=target/'bin/python'
        environment=dict(PATH=os.defpath,LANG='C.UTF-8',TMPDIR=str(temporary),PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_INPUT='1')
        for index,args in enumerate(commands):
            stage=('download','install')[index]
            (out/'state.json').write_text(json.dumps(dict(status='running',stage=stage,supervisor_pid=os.getpid())))
            with (out/(stage+'.log')).open('xb') as log:
                process=subprocess.Popen([str(python),*args],stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                last_progress=0
                while process.poll() is None:
                    elapsed=time.monotonic()-started
                    counts=dict(wheel_bytes=size(wheels),environment_bytes=size(target),temporary_bytes=size(temporary))
                    if elapsed>caps['wall_seconds'] or any(counts[k]>caps[k] for k in counts):
                        raise RuntimeError('bootstrap resource cap exceeded')
                    if elapsed-last_progress>=30:
                        print(json.dumps(dict(stage=stage,elapsed_seconds=round(elapsed,1),**counts)),flush=True)
                        last_progress=elapsed
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError('bootstrap '+stage+' failed with exit '+str(process.returncode))
                process=None
        report=json.loads((out/'install_report.json').read_text())
        wheel_hashes={}
        for path in wheels.glob('*.whl'):
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            wheel_hashes[path.name]=digest.hexdigest()
        (out/'wheel_hashes.json').write_text(json.dumps(wheel_hashes,indent=2))
        rows=[]
        for row in report['install']:
            rows.append(row['metadata']['name']+'=='+row['metadata']['version']+' --hash=sha256:'+row['download_info']['archive_info']['hashes']['sha256'])
        (out/'requirements.lock').write_text('--index-url https://download.pytorch.org/whl/cu128\n'+'\n'.join(sorted(rows,key=str.lower))+'\n')
        result=dict(status='complete',stage=stage,elapsed_seconds=time.monotonic()-started,wheel_bytes=size(wheels),environment_bytes=size(target),temporary_bytes=size(temporary),packages=len(rows),kind='dependency_install_only_no_cuda_or_model_validation')
        if result['elapsed_seconds']>caps['wall_seconds'] or any(result[k]>caps[k] for k in ('wheel_bytes','environment_bytes','temporary_bytes')):
            raise RuntimeError('post-install resource cap exceeded')
        (out/'state.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return result
    except BaseException as exc:
        if process and process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        (out/'state.json').write_text(json.dumps(dict(status='failed',stage=stage,error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started),indent=2))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    run(args.target,args.out)

"""Owned, bounded loopback vLLM service and three real basic task probes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from .artifacts import write_json

WALL_SECONDS=600
GPU_WATCHDOG_MIB=36*1024
KEY_ENV='OFFICEGYM_LOCAL_PROBE_KEY'


def request(endpoint,key,body=None,timeout=5):
    data=None if body is None else json.dumps(body).encode()
    headers={'Authorization':'Bearer '+key}
    if data is not None:headers['Content-Type']='application/json'
    with urlopen(Request(endpoint,data=data,headers=headers),timeout=timeout) as response:
        raw=response.read(4*1024*1024+1)
    if len(raw)>4*1024*1024:raise ValueError('model response frame cap')
    return json.loads(raw)


def worker(fetch_root,out):
    from officegym.training.fetch_model import validate_plan,check_file
    from officegym.tasks.generate import generate
    from officegym.eval.configured import evaluate_instance
    from officegym.harness.policy import ChatPolicy,render_messages
    from officegym.harness.tokenizer import PromptTokenizer
    from officegym.tools.schema import tool_schemas
    if importlib.metadata.version('vllm')!='0.11.0':
        raise ValueError('locked vLLM 0.11.0 required')
    source=Path(fetch_root);root=Path(out)
    plan=json.loads((source/'plan.json').read_text());validate_plan(plan)
    accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():
        raise ValueError('accepted immutable model fetch required')
    model_path=source/'model'
    for item in plan['files']:
        if not check_file(model_path/item['name'],item):raise ValueError('model bytes differ before service launch')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
    name='officegym-base-'+plan['revision'][:12]
    endpoint='http://127.0.0.1:'+str(port)
    args=[sys.executable,'-m','officegym.eval.serial_vllm','--model',str(model_path),'--served-model-name',name,'--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len','8192','--max-num-seqs','1','--gpu-memory-utilization','0.35','--enforce-eager','--enable-auto-tool-choice','--tool-call-parser','hermes']
    from .serial_vllm import ADAPTER_VERSION
    write_json(root/'server_command.json',dict(adapter=ADAPTER_VERSION,argv=args,api_key_env='VLLM_API_KEY',client_api_key_env=KEY_ENV))
    environment=dict(os.environ,VLLM_API_KEY=os.environ[KEY_ENV],VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_CACHE_ROOT=str(root/'vllm_cache'),TORCHINDUCTOR_CACHE_DIR=str(root/'torch_cache'))
    server=None
    try:
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=environment)
            write_json(root/'server_process.json',dict(pid=server.pid,port=port))
            startup=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError('vLLM service exited during startup')
                if time.monotonic()-startup>240:raise TimeoutError('vLLM startup cap')
                try:
                    models=request(endpoint+'/v1/models',os.environ[KEY_ENV],timeout=2)
                    if name not in [m['id'] for m in models['data']]:raise ValueError('served identity differs')
                    break
                except (OSError,ValueError,KeyError):time.sleep(.5)
            write_json(root/'models.json',models)
            tokenizer=PromptTokenizer.load(str(model_path),plan['revision'])
            policy=ChatPolicy(endpoint+'/v1/chat/completions',name,plan['revision'],api_key_env=KEY_ENV,max_output_tokens=128)
            messages=[dict(role='system',content='Use provided tools.'),dict(role='user',content='Read the board with board.read and empty arguments.')]
            schemas=[dict(type='function',function=dict(name=t['name'],description=t['description'],parameters=t['inputSchema'])) for t in tool_schemas()]
            native_ids=tokenizer.token_ids(messages,tool_schemas())
            actual_tokens=request(endpoint+'/tokenize',os.environ[KEY_ENV],dict(model=name,messages=render_messages(messages),tools=schemas,add_generation_prompt=True))
            write_json(root/'tokenization.json',dict(native_ids=native_ids,server=actual_tokens))
            if native_ids!=actual_tokens['tokens']:raise ValueError('server token IDs differ from native prompt')
            history=messages+[dict(role='assistant',tool_call=dict(id='history',name='people.search',arguments=dict(query='张三',limit=2))),dict(role='tool',tool_call_id='history',content='{"items":[]}')]
            history_ids=tokenizer.token_ids(history,tool_schemas())
            history_server=request(endpoint+'/tokenize',os.environ[KEY_ENV],dict(model=name,messages=render_messages(history),tools=schemas,add_generation_prompt=True))
            write_json(root/'history_tokenization.json',dict(native_ids=history_ids,server=history_server))
            if history_ids!=history_server['tokens']:raise ValueError('multi-turn server IDs differ from native prompt')
            expected=len(native_ids)
            turn=policy.generate(messages,tool_schemas(),dict(generated_tokens=128,wall_seconds=60))
            write_json(root/'protocol.json',dict(raw=turn.raw,tool=turn.tool,arguments=turn.arguments,prompt_tokens_expected=expected,prompt_tokens_measured=turn.input_tokens,generated_tokens=turn.output_tokens))
            if turn.tool!='board.read' or turn.arguments!={} or turn.input_tokens!=expected:
                raise ValueError('model API parsing/tokenization protocol mismatch')
            manifest=generate(root/'tasks',templates=['T01','T02','T03'],count_per_template=1)
            records=[]
            for entry in manifest['instances']:
                config=dict(endpoint=endpoint+'/v1/chat/completions',model=name,model_version=plan['revision'],api_key_env=KEY_ENV,tokenizer_path=str(model_path),tokenizer_revision=plan['revision'],mode='full',budget=dict(tool_calls=24,generated_tokens=4096,wall_seconds=120,parse_retries=2),max_context_tokens=8192,max_output_tokens=256,compression_trigger_tokens=7168,temperature=0)
                config_path=root/(entry['instance_id']+'_config.json');write_json(config_path,config)
                result=evaluate_instance(root/'tasks'/entry['path'],root/'runs'/entry['instance_id'],config_path)
                records.append(dict(instance_id=entry['instance_id'],termination=result['termination'],grade=result['grade'],infrastructure_error=result['infrastructure_error']))
                write_json(root/'episodes.json',records)
            write_json(root/'summary.json',dict(kind='real_base_model_feasibility_not_frozen_evaluation',model_revision=plan['revision'],episodes=len(records),successful=sum(r['grade']['full_success'] for r in records),infrastructure_errors=sum(bool(r['infrastructure_error']) or r['termination']['state']=='INFRA_ERROR' for r in records),records=records,optimizer_updates=0))
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)


def run(fetch_root,out):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'executed_worker_source.py').write_bytes(Path(__file__).read_bytes())
    from . import serial_vllm
    (root/'executed_adapter_source.py').write_bytes(Path(serial_vllm.__file__).read_bytes())
    write_json(root/'plan.json',dict(adapter_sha256=hashlib.sha256(Path(serial_vllm.__file__).read_bytes()).hexdigest(),kind='local_vllm_feasibility_plan',max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,gpu_memory_utilization=.35,gpu_observed_watchdog_mib=GPU_WATCHDOG_MIB,max_api_cost_usd=0,api_key_env=KEY_ENV,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    occupancy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if occupancy.stdout.strip():raise RuntimeError('GPU must be idle; existing processes preserved')
    environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    environment[KEY_ENV]=secrets.token_urlsafe(32)
    started=time.monotonic();peak=0
    with (root/'worker.stdout').open('w') as stdout,(root/'worker.stderr').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.eval.vllm_probe','--worker','--fetch-root',str(Path(fetch_root).resolve()),'--out',str(root)],env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
        write_json(root/'owned_process.json',dict(pid=child.pid,pgid=child.pid))
        try:
            while child.poll() is None:
                if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('vLLM probe overall wall cap')
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                owned=0
                for line in query.stdout.splitlines():
                    pid,memory=[piece.strip() for piece in line.split(',')]
                    try:
                        if os.getpgid(int(pid))==child.pid:owned+=int(memory)
                    except ProcessLookupError:pass
                peak=max(peak,owned)
                if owned>GPU_WATCHDOG_MIB:raise RuntimeError('owned vLLM observed VRAM watchdog')
                time.sleep(1)
            if child.returncode:raise RuntimeError('vLLM worker failed; inspect retained logs')
            if not (root/'summary.json').exists():raise RuntimeError('missing model episode summary')
            write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak,worker_returncode=child.returncode))
        except BaseException as exc:
            write_json(root/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak))
            raise
        finally:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            # Worker exit does not imply every model child has exited.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    return json.loads((root/'summary.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:worker(Path(args.fetch_root),Path(args.out))
    else:print(json.dumps(run(args.fetch_root,args.out)))

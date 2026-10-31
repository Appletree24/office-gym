"""Bounded public Qwen checkpoint fetch with immutable Hub and content identities."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from officegym.eval.artifacts import write_json

MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
SMALL={'LICENSE','config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'}


def allowed(name):
    return name in SMALL or re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors',name) is not None


def digests(path,expected_size):
    sha=hashlib.sha256()
    git=hashlib.sha1(b'blob '+str(expected_size).encode()+b'\0')
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha.update(block);git.update(block)
    return sha,git


def check_file(path,item):
    if not Path(path).is_file() or Path(path).stat().st_size!=item['size']:
        return False
    sha,git=digests(path,item['size'])
    return (sha if item['algorithm']=='sha256' else git).hexdigest()==item['digest']


def validate_caps(max_bytes,max_wall_seconds):
    if type(max_bytes) is not int or not 0<max_bytes<=10*1024**3:
        raise ValueError('model byte cap must be within 10 GiB')
    wall=max_wall_seconds
    if type(wall) not in (int,float) or not math.isfinite(wall) or not 0<wall<=1800:
        raise ValueError('finite fetch wall cap within 1800 seconds required')


def validate_plan(plan):
    if plan.get('model')!=MODEL or not isinstance(plan.get('revision'),str) or not re.fullmatch('[0-9a-f]{40}',plan['revision']):
        raise ValueError('fixed public model and full revision required')
    validate_caps(plan.get('max_bytes'),plan.get('max_wall_seconds'))
    files=plan.get('files',[])
    names=[f['name'] for f in files]
    if not SMALL<=set(names) or len(names)!=len(set(names)) or not any(name.endswith('.safetensors') for name in names):
        raise ValueError('complete unique checkpoint file list required')
    for item in files:
        if not isinstance(item['name'],str) or not allowed(item['name']) or type(item['size']) is not int or item['size']<=0:
            raise ValueError('invalid checkpoint file path/size')
        length={'sha256':64,'sha1_git_blob':40}.get(item['algorithm'])
        if length is None or not isinstance(item['digest'],str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',item['digest']):
            raise ValueError('invalid checkpoint content identity')
    if sum(f['size'] for f in files)>plan['max_bytes']:
        raise ValueError('declared model exceeds download byte cap')


def fetch_file(item,directory,revision,progress):
    destination=directory/item['name']
    partial=directory/(item['name']+'.part')
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError('checkpoint paths cannot be symlinks')
    if destination.exists():
        if not check_file(destination,item):
            raise ValueError('existing checkpoint file failed immutable identity')
        return
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>item['size']:
        raise ValueError('partial checkpoint exceeds expected size')
    if offset<item['size']:
        # Resolve afresh to avoid a cached expired signed CDN redirect. Only the
        # immutable canonical model path is persisted; signed URLs are not logged.
        url='https://huggingface.co/'+MODEL+'/resolve/'+revision+'/'+item['name']+'?download=true&officegym_request='+uuid.uuid4().hex
        request=urllib.request.Request(url,headers={'Range':'bytes='+str(offset)+'-'} if offset else {})
        with urllib.request.urlopen(request,timeout=20) as response:
            if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')):
                raise ValueError('server did not honor immutable range resume')
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                while True:
                    block=response.read(65536)
                    if not block:
                        break
                    if offset+len(block)>item['size']:
                        raise ValueError('response exceeds declared checkpoint size')
                    stream.write(block);offset+=len(block)
                    progress(item['name'],offset)
                stream.flush();os.fsync(stream.fileno())
    if not check_file(partial,item):
        raise ValueError('downloaded checkpoint size/content digest mismatch')
    partial.replace(destination)


def worker(plan_path,expected_sha):
    path=Path(plan_path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('fetch plan changed')
    plan=json.loads(path.read_text());validate_plan(plan)
    root=path.parent;directory=root/'model';directory.mkdir(exist_ok=True)
    started=time.monotonic();last=0
    def progress(name,offset):
        nonlocal last
        elapsed=time.monotonic()-started
        if elapsed>plan['max_wall_seconds']:
            raise TimeoutError('checkpoint stream wall cap')
        if elapsed-last>=5:
            write_json(root/'progress.json',dict(status='running',file=name,file_bytes=offset,elapsed_seconds=elapsed,pid=os.getpid()))
            last=elapsed
    for item in plan['files']:
        progress(item['name'],0)
        seed=Path(plan['seed_tokenizer'])/item['name'] if plan.get('seed_tokenizer') and item['name'] in SMALL else None
        destination=directory/item['name']
        if not destination.exists() and not (directory/(item['name']+'.part')).exists() and seed and check_file(seed,item):
            shutil.copyfile(seed,destination)
        fetch_file(item,directory,plan['revision'],progress)
        write_json(root/'progress.json',dict(status='verified_file',file=item['name'],file_bytes=item['size'],elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
    index=json.loads((directory/'model.safetensors.index.json').read_text())
    shards={f['name'] for f in plan['files'] if f['name'].endswith('.safetensors')}
    if set(index['weight_map'].values())!=shards:
        raise ValueError('checkpoint index does not match downloaded shards')
    write_json(root/'verified.json',dict(status='complete',model=MODEL,revision=plan['revision'],files=plan['files'],total_bytes=sum(f['size'] for f in plan['files']),model_path=str(directory),kind='verified_download_not_model_compatibility',model_inference_calls=0,optimizer_updates=0))


def run(out,*,revision=REVISION,max_bytes=10*1024**3,max_wall_seconds=1800,seed_tokenizer=None,resume=False):
    validate_caps(max_bytes,max_wall_seconds)
    coordinator_started=time.monotonic()
    root=Path(out).resolve()
    if resume:
        plan_path=root/'plan.json';plan=json.loads(plan_path.read_text());validate_plan(plan)
        if (root/'result.json').exists() and json.loads((root/'result.json').read_text()).get('status')=='complete':
            raise ValueError('already complete; reuse verified model rather than restart')
        if max_bytes!=plan['max_bytes'] or max_wall_seconds!=plan['max_wall_seconds'] or revision!=plan['revision']:
            raise ValueError('resume cannot change source or caps')
    else:
        if root.exists():
            raise ValueError('fresh output or explicit resume required')
        if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('full immutable revision required')
        root.mkdir(parents=True)
        url='https://huggingface.co/api/models/'+MODEL+'/tree/'+revision+'?recursive=true&expand=true'
        try:
            metadata_deadline=coordinator_started+min(20,max_wall_seconds)
            with urllib.request.urlopen(url,timeout=min(20,max_wall_seconds)) as response:
                chunks=[];count=0
                while True:
                    if time.monotonic()>=metadata_deadline:
                        raise TimeoutError('checkpoint metadata wall budget exhausted')
                    block=response.read1(65536)
                    if not block:
                        break
                    count+=len(block)
                    if count>2*1024*1024:
                        raise ValueError('model metadata exceeds cap')
                    chunks.append(block)
            body=b''.join(chunks)
        except BaseException as exc:
            write_json(root/'preflight_failure.json',dict(error_type=type(exc).__name__,phase='metadata',wall_seconds=time.monotonic()-coordinator_started))
            raise
        metadata=json.loads(body);files=[]
        for row in metadata:
            name=row['path']
            if row['type']=='file' and allowed(name):
                lfs=row.get('lfs')
                files.append(dict(name=name,size=row['size'],algorithm='sha256' if lfs else 'sha1_git_blob',digest=lfs['oid'] if lfs else row['oid']))
        files.sort(key=lambda f:(f['name'].endswith('.safetensors'),f['size']))
        plan=dict(model=MODEL,revision=revision,files=files,max_bytes=max_bytes,max_wall_seconds=max_wall_seconds,seed_tokenizer=str(Path(seed_tokenizer).resolve()) if seed_tokenizer else None,metadata_sha256=hashlib.sha256(body).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='public_checkpoint_fetch_plan_no_gpu_or_api_model_job')
        validate_plan(plan)
        if shutil.disk_usage(root).free<sum(f['size'] for f in files)+10*1024**3:
            raise ValueError('insufficient disk for checkpoint plus reserve')
        (root/'hub_metadata.json').write_bytes(body)
        plan_path=root/'plan.json';write_json(plan_path,plan)
    attempts=root/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/uuid.uuid4().hex;attempt.mkdir()
    digest=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    source=Path(__file__).read_bytes()
    (attempt/'worker_source.py').write_bytes(source)
    write_json(attempt/'plan_binding.json',dict(plan_sha256=digest,worker_source_sha256=hashlib.sha256(source).hexdigest(),coordinator_pid=os.getpid(),max_wall_seconds=max_wall_seconds))
    environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    # Trusted fetch worker keeps configured network routing, never records values.
    network_keys=('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','NO_PROXY','https_proxy','http_proxy','all_proxy','no_proxy','SSL_CERT_FILE','SSL_CERT_DIR')
    for key in network_keys:
        if os.environ.get(key):
            environment[key]=os.environ[key]
    write_json(attempt/'network_environment.json',dict(forwarded_keys=[key for key in network_keys if key in environment]))
    started=coordinator_started
    with (attempt/'stdout.log').open('w') as stdout,(attempt/'stderr.log').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.training.fetch_model','--worker','--plan',str(plan_path),'--plan-sha256',digest],env=environment,stdout=stdout,stderr=stderr)
        write_json(attempt/'process.json',dict(coordinator_pid=os.getpid(),worker_pid=child.pid))
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>max_wall_seconds:
                    raise TimeoutError('checkpoint subprocess wall cap')
                if shutil.disk_usage(root).free<10*1024**3:
                    raise RuntimeError('checkpoint disk reserve exhausted')
                time.sleep(1)
            if child.returncode:
                raise RuntimeError('checkpoint worker failed; inspect retained attempt stderr')
            if not (root/'verified.json').exists():
                raise RuntimeError('checkpoint worker returned without verification')
            elapsed=time.monotonic()-started
            if elapsed>max_wall_seconds:
                raise TimeoutError('checkpoint completed outside wall acceptance cap')
            write_json(attempt/'result.json',dict(status='complete',wall_seconds=elapsed))
            write_json(root/'result.json',dict(status='complete',attempt=str(attempt),plan_sha256=digest,wall_seconds=elapsed,resource_accepted=True,kind='verified_download_not_model_compatibility'))
        except BaseException as exc:
            if child.poll() is None:
                child.kill();child.wait(timeout=5)
            write_json(attempt/'result.json',dict(status='failed',error_type=type(exc).__name__,wall_seconds=time.monotonic()-started))
            raise
    return json.loads((root/'verified.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    parser.add_argument('--revision',default=REVISION)
    parser.add_argument('--max-bytes',type=int,default=10*1024**3)
    parser.add_argument('--max-wall-seconds',type=float,default=1800)
    parser.add_argument('--seed-tokenizer')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--plan',help=argparse.SUPPRESS)
    parser.add_argument('--plan-sha256',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.plan,args.plan_sha256)
    elif args.out:
        result=run(args.out,revision=args.revision,max_bytes=args.max_bytes,max_wall_seconds=args.max_wall_seconds,seed_tokenizer=args.seed_tokenizer,resume=args.resume)
        print(json.dumps(dict(status=result['status'],total_bytes=result['total_bytes'],revision=result['revision'])))
    else:
        parser.error('--out required')

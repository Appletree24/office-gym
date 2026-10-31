"""One bounded, offline pretrained Qwen tool-generation compatibility probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .artifacts import write_json

WALL_CAP=120
ALLOCATOR_CAP=16*1024**3


def parse_tool(text):
    calls=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if len(calls)!=1 or text.count('<tool_call>')!=1 or text.count('</tool_call>')!=1:
        raise ValueError('exactly one native tool-call block required')
    value=json.loads(calls[0])
    if not isinstance(value,dict) or set(value)!={'name','arguments'} or not isinstance(value['name'],str) or not isinstance(value['arguments'],dict):
        raise ValueError('native function name and object arguments required')
    from officegym.tools.schema import validate,WRITES
    validate(value['name'],value['arguments'],'trusted_probe_validation' if value['name'] in WRITES else None)
    return value


def worker(fetch_root,out):
    from officegym.training.fetch_model import validate_plan,check_file
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from officegym.harness.tokenizer import PromptTokenizer
    from officegym.harness.policy import render_messages
    from officegym.tools.schema import tool_schemas,TOOL_SCHEMA_VERSION
    root=Path(fetch_root)
    plan=json.loads((root/'plan.json').read_text());validate_plan(plan)
    accepted=json.loads((root/'result.json').read_text())
    if accepted.get('status')!='complete' or not accepted.get('resource_accepted') or accepted['plan_sha256']!=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest():
        raise ValueError('complete resource-accepted checkpoint fetch required')
    model_path=root/'model'
    for item in plan['files']:
        if not check_file(model_path/item['name'],item):
            raise ValueError('checkpoint content changed before loading')
    if torch.__version__!='2.8.0+cu128' or transformers.__version__!='4.57.1' or not torch.cuda.is_available():
        raise ValueError('locked native model runtime required')
    properties=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP/properties.total_memory,device=0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(2026)
    prompt=PromptTokenizer.load(str(model_path),plan['revision'])
    messages=[dict(role='system',content='Use the provided tools. Do not invent tool observations.'),dict(role='user',content='Read the board now by calling board.read with an empty arguments object. Return exactly one tool call.')]
    tools=tool_schemas()
    schemas=[dict(type='function',function=dict(name=t['name'],description=t['description'],parameters=t['inputSchema'])) for t in tools]
    ids=prompt.token_ids(messages,tools)
    if len(ids)>4096:
        raise ValueError('protocol probe prompt exceeds frozen 4096-token cap')
    write_json(out/'prompt.json',dict(messages=messages,tools=tools,input_ids=ids,tokenizer=prompt.manifest()))
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to('cuda:0').eval()
    if sum(p.numel() for p in model.parameters())!=4022468096:
        raise ValueError('target 4B model architecture differs')
    input_ids=torch.tensor([ids],device='cuda')
    with torch.inference_mode():
        output=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),max_new_tokens=128,do_sample=False,temperature=None,top_p=None,top_k=None)
    torch.cuda.synchronize()
    generated=output[0,len(ids):].tolist()
    raw=prompt.tokenizer.decode(generated,skip_special_tokens=False)
    parse_error=None
    parsed=None
    try:
        parsed=parse_tool(raw)
    except Exception as exc:
        parse_error=type(exc).__name__+': '+str(exc)
    for item in plan['files']:
        if not check_file(model_path/item['name'],item):
            raise ValueError('checkpoint content changed during inference')
    peak=torch.cuda.max_memory_allocated()
    report=dict(kind='pretrained_qwen_single_tool_protocol_not_task_evaluation',model_revision=plan['revision'],torch=torch.__version__,transformers=transformers.__version__,tool_schema_version=TOOL_SCHEMA_VERSION,parameters=sum(p.numel() for p in model.parameters()),input_tokens=len(ids),generated_tokens=len(generated),generated_ids=generated,raw_output=raw,parsed_tool=parsed,parse_error=parse_error,tool_call_passed=parsed==dict(name='board.read',arguments={}),model_load_passed=True,pretrained_inference_executed=True,model_inference_calls=1,optimizer_updates=0,peak_allocated_bytes=peak,allocated_byte_cap=ALLOCATOR_CAP,model_load_and_generation_wall_seconds=time.monotonic()-started)
    write_json(out/'raw.json',report)


def run(fetch_root,out):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    fetch_root=Path(fetch_root).resolve()
    plan=dict(kind='one_local_model_probe_plan',model_fetch_root=str(fetch_root),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),max_wall_seconds=WALL_CAP,max_gpu_hours=WALL_CAP/3600,max_allocated_bytes=ALLOCATOR_CAP,max_input_tokens=4096,max_generated_tokens=128,max_api_cost_usd=0,optimizer_updates=0)
    write_json(root/'plan.json',plan)
    try:
        occupancy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
        if occupancy.stdout.strip():
            raise RuntimeError('model probe requires idle GPU; existing processes preserved')
        environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        process=subprocess.run([sys.executable,'-m','officegym.eval.model_probe','--worker','--fetch-root',str(fetch_root),'--out',str(root)],env=environment,capture_output=True,text=True,timeout=max(.01,WALL_CAP-(time.monotonic()-started)))
        (root/'worker.stdout').write_text(process.stdout);(root/'worker.stderr').write_text(process.stderr)
        if process.returncode:
            raise RuntimeError('pretrained model worker failed; inspect retained stderr')
        raw=json.loads((root/'raw.json').read_text())
        elapsed=time.monotonic()-started
        accepted=raw['peak_allocated_bytes']<=ALLOCATOR_CAP and elapsed<=WALL_CAP and raw['generated_tokens']<=128
        report=dict(raw,wall_seconds=elapsed,max_wall_seconds=WALL_CAP,resource_accepted=accepted)
        write_json(root/'report.json',report)
        if not accepted or not raw['tool_call_passed']:
            raise ValueError('model protocol/resource checks did not pass')
        return report
    except BaseException as exc:
        if isinstance(exc,subprocess.TimeoutExpired):
            (root/'worker.stdout').write_bytes(exc.stdout or b'');(root/'worker.stderr').write_bytes(exc.stderr or b'')
        write_json(root/'failure.json',dict(error_type=type(exc).__name__,wall_seconds=time.monotonic()-started,complete_model_task_evaluation=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(Path(args.fetch_root),Path(args.out))
    else:
        print(json.dumps(run(args.fetch_root,args.out)))

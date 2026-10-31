"""Real merge/numerical probe from a labelled optimizer-backend checkpoint.

Does not register a formal SFT export or bypass teacher/tool-equivalence gates.
"""
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

WALL_SECONDS=480
ALLOCATOR_BYTES=24*1024**3
WATCHDOG_MIB=32*1024


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(fetch_root,backend,out,profile):
    from .fetch_model import validate_plan,check_file
    from .checkpoints import checkpoint_fingerprint
    from . import merge,equivalence
    from officegym.harness.tokenizer import PromptTokenizer
    from officegym.tools.schema import tool_schemas
    source=Path(fetch_root);backend=Path(backend);root=Path(out)
    if profile not in ('bf16','fp32'):raise ValueError('explicit bf16/fp32 profile required')
    dtype='bfloat16' if profile=='bf16' else 'float32'
    atol=.25 if profile=='bf16' else 1e-4
    export_cap=(10 if profile=='bf16' else 20)*1024**3
    fetch=json.loads((source/'plan.json').read_text());validate_plan(fetch)
    accepted=json.loads((source/'result.json').read_text())
    if accepted.get('status')!='complete' or not accepted.get('resource_accepted') or accepted['plan_sha256']!=sha(source/'plan.json'):raise ValueError('verified model fetch required')
    for item in fetch['files']:
        if not check_file(source/'model'/item['name'],item):raise ValueError('base source bytes changed')
    report=json.loads((backend/'report.json').read_text())
    if report.get('kind')!='authored_protocol_fixture_real_optimizer_not_formal_sft' or report.get('passed') is not True or report.get('optimizer_updates')!=2 or report.get('source_model_revision')!=fetch['revision']:raise ValueError('successful labelled backend probe source required')
    adapter=backend/'trainer/checkpoint-2'
    if sha(adapter/'adapter_model.safetensors')!=report['checkpoint']['files_sha256']['adapter_model.safetensors']:raise ValueError('checkpoint adapter differs from verified update')
    base=checkpoint_fingerprint(source/'model')
    plan=dict(base_model_path=str((source/'model').resolve()),base_revision=fetch['revision'],base_checkpoint_sha256=base['checkpoint_sha256'],adapter_path=str(adapter.resolve()),training_events=str((backend/'training_events.jsonl').resolve()),optimizer_step=2,locked_packages=dict(torch='2.8.0+cu128',transformers='4.57.1',peft='0.17.1'),device='cuda',dtype=dtype,max_allocated_bytes=ALLOCATOR_BYTES,max_wall_seconds=240,max_shard_size_bytes=2*1024**3,max_export_bytes=export_cap)
    plan_path=root/'merge_plan.json';write_json(plan_path,plan)
    tokenizer=PromptTokenizer.load(str(source/'model'),fetch['revision'])
    probes=[]
    for case,prompt in [('read_board','Call board.read with empty arguments.'),('unicode_search','Call people.search with query 中文 Ω.')]:
        ids=tokenizer.token_ids([dict(role='system',content='Use provided tools. Return exactly one tool call.'),dict(role='user',content=prompt)],tool_schemas())
        probes.append(dict(case_id=case,input_ids=ids))
    probes_path=root/'fixed_probes.json';write_json(probes_path,probes)
    write_json(root/'source_identity.json',dict(fetch_plan_sha256=sha(source/'plan.json'),backend_report_sha256=sha(backend/'report.json'),tokenizer=tokenizer.manifest(),formal_sft=False))
    status=merge.merge(plan_path,root/'merge',own_process_group=False)
    if not status['merge_completed']:raise RuntimeError('real merge failed; retained logs')
    result_path=root/'merge/merge_result.json'
    numeric=dict(merge_plan=str(plan_path),merge_plan_sha256=sha(plan_path),merge_result=str(result_path),merge_result_sha256=sha(result_path),probes=str(probes_path),probes_sha256=sha(probes_path),logit_atol=atol,min_generation_match_rate=1.,max_cases=2,max_input_tokens=4096,max_new_tokens=64,max_wall_seconds=240,max_probe_logits_elements=640000000)
    numeric_path=root/'numeric_plan.json';write_json(numeric_path,numeric)
    status=equivalence.verify(numeric_path,root/'numeric',own_process_group=False)
    numeric_result=root/'numeric/numeric_equivalence.json'
    if status['exit_code'] not in (0,1) or status['wall_budget_exhausted'] or not numeric_result.exists():raise RuntimeError('real numerical check failed; retained logs')
    comparison=json.loads(numeric_result.read_text())
    write_json(root/'summary.json',dict(kind='backend_adapter_merge_numeric_probe_not_formal_sft_export',profile=profile,numeric_worker_exit_code=status['exit_code'],passed_numeric=comparison['passed_numeric'],full_equivalence_passed=False,formal_sft=False,teacher_samples=0,tool_equivalence='not_checked',comparisons=comparison['comparisons'],generation_match_rate=comparison['generation_match_rate'],exported_checkpoint_sha256=comparison['exported_checkpoint_sha256']))


def run(fetch_root,backend,out,profile='bf16'):
    if profile not in ('bf16','fp32'):raise ValueError('explicit bf16/fp32 profile required')
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    from . import merge,equivalence,checkpoints,launch
    for module in (sys.modules[__name__],merge,equivalence,checkpoints,launch):
        path=Path(module.__file__);(root/('executed_'+path.name)).write_bytes(path.read_bytes())
    write_json(root/'plan.json',dict(kind='bounded_backend_export_probe',max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,max_allocated_bytes=ALLOCATOR_BYTES,gpu_watchdog_mib=WATCHDOG_MIB,max_api_cost_usd=0,formal_sft=False,profile=profile,logit_atol=.25 if profile=='bf16' else 1e-4,min_generation_match_rate=1.,max_new_tokens=64))
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if idle.stdout.strip():
        write_json(root/'failure.json',dict(stage='idle_preflight',resource_accepted=False,error='GPU occupied; existing jobs preserved'))
        raise ValueError('GPU must be idle; existing jobs preserved')
    env=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(Path(__file__).resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started=time.monotonic();peak=0
    with (root/'worker.stdout').open('w') as stdout,(root/'worker.stderr').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.training.export_probe','--worker','--profile',profile,'--fetch-root',str(Path(fetch_root).resolve()),'--backend',str(Path(backend).resolve()),'--out',str(root)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        write_json(root/'process.json',dict(pid=child.pid,pgid=child.pid))
        try:
            while child.poll() is None:
                if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('export probe wall cap')
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                used=0
                for line in query.stdout.splitlines():
                    pid,memory=line.split(',')
                    try:
                        if os.getpgid(int(pid))==child.pid:used+=int(memory)
                    except ProcessLookupError:pass
                peak=max(peak,used)
                if used>WATCHDOG_MIB:raise RuntimeError('owned export GPU watchdog')
                time.sleep(1)
            if child.returncode:raise RuntimeError('export probe worker failed; retained logs')
            summary=json.loads((root/'summary.json').read_text())
            write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak))
            return summary
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--backend',required=True);parser.add_argument('--out',required=True);parser.add_argument('--profile',choices=('bf16','fp32'),default='bf16');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.worker:
        worker(Path(args.fetch_root),Path(args.backend),Path(args.out),args.profile)
        return 0
    result=run(args.fetch_root,args.backend,args.out,args.profile)
    print(json.dumps(result))
    return 0 if result['passed_numeric'] else 1


if __name__=='__main__':raise SystemExit(main())

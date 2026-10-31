"""Bounded real-weight TRL optimizer probe using authored protocol fixtures only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from officegym.eval.artifacts import write_json

WALL_SECONDS=360
ALLOCATOR_BYTES=40*1024**3
WATCHDOG_MIB=48*1024


def worker(fetch_root,out):
    import torch
    from transformers import TrainerCallback
    from officegym.training.fetch_model import validate_plan,check_file
    from officegym.harness.tokenizer import PromptTokenizer
    from officegym.tools.schema import tool_schemas
    from .tokenizer import SFTTokenizer
    from .sft import make_trainer
    from .evidence import TrainingEvidence
    root=Path(out);source=Path(fetch_root)
    fetch=json.loads((source/'plan.json').read_text());validate_plan(fetch)
    accepted=json.loads((source/'result.json').read_text())
    if not accepted.get('resource_accepted') or accepted.get('status')!='complete' or accepted['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():raise ValueError('verified fixed model fetch required')
    def check_weights():
        for item in fetch['files']:
            if not check_file(source/'model'/item['name'],item):raise ValueError('source model bytes changed')
    check_weights()
    locked={'torch':'2.8.0+cu128','transformers':'4.57.1','trl':'0.24.0','peft':'0.17.1','datasets':'4.3.0','accelerate':'1.10.1'}
    if any(importlib.metadata.version(k)!=v for k,v in locked.items()) or not torch.cuda.is_available():raise ValueError('locked CUDA training runtime required')
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(2026)
    tokenizer=PromptTokenizer.load(str(source/'model'),fetch['revision']).tokenizer
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    alignment=SFTTokenizer(tokenizer);rows=[]
    for query in ('owner','中文 Ω'):
        messages=[dict(role='system',content='Use tools; observations are evidence.'),dict(role='user',content='Read the board and search for '+query),dict(role='assistant',tool_call=dict(id='one',name='board.read',arguments={})),dict(role='tool',tool_call_id='one',content='{"ok":true,"data":{"content":{},"version":1}}'),dict(role='assistant',tool_call=dict(id='two',name='people.search',arguments=dict(query=query))),dict(role='tool',tool_call_id='two',content='{"ok":true,"data":{"items":[]}}')]
        encoded=alignment(messages,tool_schemas());encoded['labels']=[token if bit else -100 for token,bit in zip(encoded['input_ids'],encoded['assistant_mask'])];rows.append(encoded)
    (root/'protocol_samples.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    write_json(root/'mask_manifest.json',alignment.mask_manifest())
    plan=dict(model_path=str(source/'model'),model_revision=fetch['revision'],max_steps=2,max_wall_seconds=WALL_SECONDS,locked_packages=locked,max_sequence_tokens=4096,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-4,lora=dict(rank=16,alpha=32,dropout=.05,target_modules='all-linear'),checkpoint_interval_steps=1,gradient_checkpointing=True,seed=2026)
    write_json(root/'resolved_backend_config.json',plan)
    evidence=TrainingEvidence(root);gradients=[];losses=[]
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            finite=True;norm=0.;count=0
            for name,p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:raise ValueError('frozen base received gradients')
                if p.requires_grad and p.grad is not None:
                    finite=finite and bool(torch.isfinite(p.grad).all());norm+=p.grad.detach().float().square().sum().item();count+=1
            if not finite or norm<=0 or count==0:raise ValueError('missing finite nonzero adapter gradients')
            row=dict(step=state.global_step+1,gradient_squared_norm=norm,gradient_tensors=count,all_finite=finite);gradients.append(row);evidence.event('gradient_check',state.global_step,row)
        def on_log(self,args,state,control,logs=None,**kwargs):evidence.event('training_log',state.global_step,logs or {})
        def on_save(self,args,state,control,**kwargs):evidence.checkpoint(root/'trainer'/('checkpoint-'+str(state.global_step)),state.global_step)
    trainer=make_trainer(plan,rows,root,tokenizer,[Audit()]);model=trainer.model
    trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    if any('lora_' not in n for n in trainable):raise ValueError('unexpected trainable base parameter')
    base_parameters=sum(p.numel() for n,p in model.named_parameters() if 'lora_' not in n)
    if base_parameters!=4022468096:raise ValueError('wrong pretrained base architecture')
    def base_fingerprint():
        digest=hashlib.sha256()
        for n,p in model.named_parameters():
            if not p.requires_grad:
                values=p.detach().reshape(-1);digest.update(n.encode());digest.update(torch.cat([values[:64],values[-64:]]).float().cpu().numpy().tobytes())
        return digest.hexdigest()
    base_before=base_fingerprint();original=trainer.compute_loss
    def audited_loss(model,inputs,return_outputs=False,num_items_in_batch=None):
        labels=inputs['labels'].detach().clone()
        loss,outputs=original(model,inputs,return_outputs=True,num_items_in_batch=num_items_in_batch)
        with torch.no_grad():
            targets=labels[:,1:];valid=targets!=-100
            logits=outputs.logits[:,:-1][valid].float()
            expected=torch.nn.functional.cross_entropy(logits,targets[valid])
            if not torch.isfinite(loss) or not torch.allclose(loss.detach().float(),expected,atol=1e-4,rtol=1e-4):raise ValueError('native loss differs from assistant-only shifted cross entropy')
            row=dict(loss=loss.item(),independent_assistant_only_loss=expected.item(),supervised_shifted_tokens=valid.sum().item(),masked_tokens=(~valid).sum().item());losses.append(row);evidence.event('loss_mask_check',trainer.state.global_step,row)
        return (loss,outputs) if return_outputs else loss
    trainer.compute_loss=audited_loss
    result=trainer.train();torch.cuda.synchronize();evidence.history(trainer.state.log_history)
    changed=[n for n,p in model.named_parameters() if n in trainable and not torch.equal(p.detach().cpu(),trainable[n])]
    if trainer.state.global_step!=2 or len(gradients)!=2 or len(losses)!=2 or not changed or base_fingerprint()!=base_before:raise ValueError('optimizer/base-freeze verification failed')
    trainer.save_model(str(root/'adapter'));tokenizer.save_pretrained(root/'adapter');checkpoint=evidence.checkpoint(root/'adapter',2,kind='backend_probe_adapter')
    check_weights()
    write_json(root/'report.json',dict(kind='authored_protocol_fixture_real_optimizer_not_formal_sft',passed=True,source_model_revision=fetch['revision'],base_parameters=base_parameters,trainable_parameters=sum(v.numel() for v in trainable.values()),changed_adapter_tensors=len(changed),base_sample_fingerprint_unchanged=True,source_weights_unchanged=True,optimizer_updates=2,loss_checks=losses,gradient_checks=gradients,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint=checkpoint,metrics=result.metrics,teacher_samples=0,task_success_measurements=0,export_equivalence='not_checked'))


def run(fetch_root,out):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__);(root/'executed_probe_source.py').write_bytes(source.read_bytes())
    from . import sft
    (root/'executed_sft_source.py').write_bytes(Path(sft.__file__).read_bytes())
    write_json(root/'plan.json',dict(kind='two_update_real_weight_backend_probe',max_wall_seconds=WALL_SECONDS,max_gpu_hours=WALL_SECONDS/3600,max_allocated_bytes=ALLOCATOR_BYTES,gpu_watchdog_mib=WATCHDOG_MIB,max_api_cost_usd=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),formal_sft=False))
    idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if idle.stdout.strip():raise ValueError('GPU must be idle; preserve existing jobs')
    environment=dict(PATH=os.defpath,LANG='C.UTF-8',PYTHONPATH=str(source.resolve().parents[2]),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_DISABLED='true',HF_DATASETS_OFFLINE='1')
    started=time.monotonic();peak=0
    with (root/'worker.stdout').open('w') as stdout,(root/'worker.stderr').open('w') as stderr:
        child=subprocess.Popen([sys.executable,'-m','officegym.training.backend_probe','--worker','--fetch-root',str(Path(fetch_root).resolve()),'--out',str(root)],env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
        write_json(root/'process.json',dict(pid=child.pid,pgid=child.pid))
        try:
            while child.poll() is None:
                if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('backend probe wall cap')
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                memory=0
                for line in query.stdout.splitlines():
                    pid,used=line.split(',')
                    try:
                        if os.getpgid(int(pid))==child.pid:memory+=int(used)
                    except ProcessLookupError:pass
                peak=max(peak,memory)
                if memory>WATCHDOG_MIB:raise RuntimeError('backend observed memory cap')
                time.sleep(1)
            if child.returncode:raise RuntimeError('backend worker failed; retained logs')
            report=json.loads((root/'report.json').read_text())
            if not report.get('passed') or report['peak_allocated_bytes']>ALLOCATOR_BYTES:raise ValueError('backend evidence failed')
            write_json(root/'resource_result.json',dict(status='complete',wall_seconds=time.monotonic()-started,observed_peak_gpu_mib=peak))
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
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.worker:worker(Path(args.fetch_root),Path(args.out))
    else:print(json.dumps(run(args.fetch_root,args.out)))

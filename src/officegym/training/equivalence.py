"""Bounded local fixed-token numerical checks; tool equivalence stays separate."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from officegym.eval.artifacts import write_json
from .merge import validate_merge_plan,check_packages,configure_memory
from .checkpoints import checkpoint_fingerprint
from .launch import execute_bounded


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


NUMERIC_REPORT_VERSION='numeric-v2-replayable-elementwise'


def compare_logits(left,right,atol,rtol=0):
    """Compare candidate right against reference left, like allclose(right,left)."""
    if not left or len(left)!=len(right):
        raise ValueError('unaligned nonempty probe logits')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (atol,rtol)):
        raise ValueError('finite nonnegative logit tolerances required')
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in left+right):
        return dict(finite=False,max_abs_error=None,max_scaled_error=None,violating_elements=None,elements=len(left),within_tolerance=False)
    errors=[abs(a-b) for a,b in zip(left,right)]
    bounds=[atol+rtol*abs(a) for a in left]
    if any(not math.isfinite(v) for v in errors+bounds):
        raise ValueError('nonfinite elementwise error/budget')
    violations=sum(e>b for e,b in zip(errors,bounds))
    scaled=None if any(b==0 and e>0 for e,b in zip(errors,bounds)) else max(e/b if b else 0 for e,b in zip(errors,bounds))
    return dict(finite=True,max_abs_error=max(errors),max_scaled_error=scaled,violating_elements=violations,elements=len(left),within_tolerance=violations==0)


def replay_comparisons(report,probes,vocab,atol,rtol):
    """Recompute metrics from hash-bound plain JSON, without importing Torch."""
    expected=[p['case_id'] for p in probes]
    if report.get('numerical_report_version')!=NUMERIC_REPORT_VERSION:
        raise ValueError('replayable numerical report version required')
    sides={}
    for side in ('before','after'):
        if [r['case_id'] for r in report[side]]!=expected:raise ValueError('incomplete numerical artifacts')
        sides[side]=[]
        for row,probe in zip(report[side],probes):
            path=Path(row['artifact'])
            if not path.is_absolute() or digest(path)!=row['sha256']:raise ValueError('numerical artifact hash/path differs')
            raw=json.loads(path.read_text())
            if not isinstance(raw.get('input_ids'),list) or any(type(t) is not int for t in raw['input_ids']):raise ValueError('invalid original input token IDs')
            if not isinstance(raw.get('generated_ids'),list) or not raw['generated_ids']:raise ValueError('nonempty generated token IDs required')
            if raw['input_ids']!=probe['input_ids'] or raw['generated_ids']!=row['generated_ids'] or len(raw['next_token_logits'])!=vocab:
                raise ValueError('numerical artifact IDs/vocabulary differ')
            if any(type(t) is not int or not 0<=t<vocab for t in raw['generated_ids']):raise ValueError('invalid generated token IDs')
            sides[side].append(raw)
    comparisons=[]
    for probe,left,right in zip(probes,sides['before'],sides['after']):
        comparisons.append(dict(case_id=probe['case_id'],logits=compare_logits(left['next_token_logits'],right['next_token_logits'],atol,rtol),generation_equal=left['generated_ids']==right['generated_ids']))
    if comparisons!=report['comparisons']:raise ValueError('numerical metrics differ from raw artifact replay')
    return comparisons


def replay_control(report,probes,vocab,atol,rtol,required):
    if not required:return True
    control=report.get('negative_control',{})
    if required!='disabled_adapter' or control.get('kind')!=required:raise ValueError('declared negative control missing')
    shadow=dict(report,after=control['records'],comparisons=control['comparisons'])
    rows=replay_comparisons(shadow,probes,vocab,atol,rtol)
    detected=any(not row['logits']['within_tolerance'] for row in rows)
    if control.get('detected')!=detected:raise ValueError('negative control metric differs from replay')
    return detected


def validate(plan):
    required={'merge_plan','merge_plan_sha256','merge_result','merge_result_sha256','probes','probes_sha256','logit_atol','min_generation_match_rate','max_cases','max_input_tokens','max_new_tokens','max_wall_seconds','max_probe_logits_elements'}
    if required-set(plan):
        raise ValueError('missing numerical equivalence fields')
    for field in ('max_cases','max_input_tokens','max_new_tokens','max_probe_logits_elements'):
        if type(plan[field]) is not int or plan[field]<1:
            raise ValueError('finite positive probe budget required')
    for field in ('logit_atol','min_generation_match_rate','max_wall_seconds'):
        if type(plan[field]) not in (int,float) or not math.isfinite(plan[field]):
            raise ValueError('finite numerical equivalence thresholds required')
    if plan.get('negative_control') not in (None,'disabled_adapter'):raise ValueError('unsupported negative control')
    if plan.get('attention_backend','auto') not in ('auto','math'):raise ValueError('explicit supported numerical attention backend required')
    rtol=plan.get('logit_rtol',0)
    if type(rtol) not in (int,float) or not math.isfinite(rtol) or rtol<0:raise ValueError('finite nonnegative relative logit tolerance required')
    if plan['logit_atol']<0 or not 0<=plan['min_generation_match_rate']<=1 or plan['max_wall_seconds']<=0:
        raise ValueError('invalid numerical equivalence thresholds')
    for field in ('merge_plan','merge_result','probes'):
        if not Path(plan[field]).is_absolute() or digest(plan[field])!=plan[field+'_sha256']:
            raise ValueError('numerical equivalence input hash/path mismatch')
    merge_plan=json.loads(Path(plan['merge_plan']).read_text())
    source=validate_merge_plan(merge_plan)
    merged=json.loads(Path(plan['merge_result']).read_text())
    if merged.get('merged') is not True or merged.get('training_source')!=source['training_source'] or merged.get('base_checkpoint_sha256')!=source['base_checkpoint_sha256']:
        raise ValueError('merged source identity mismatch')
    if merge_plan.get('merge_accumulation','native')=='float64' and merged.get('merge_arithmetic',{}).get('accumulation_dtype')!='float64':raise ValueError('merge arithmetic differs from the frozen plan')
    if not Path(merged['model_path']).is_absolute():
        raise ValueError('absolute merged model path required')
    if checkpoint_fingerprint(merged['model_path'])['checkpoint_sha256']!=merged['exported_checkpoint_sha256']:
        raise ValueError('merged weights changed before equivalence check')
    vocab=json.loads((Path(merged['model_path'])/'config.json').read_text()).get('vocab_size')
    if type(vocab) is not int or vocab<1:
        raise ValueError('declared model vocabulary required for probe budget')
    probes=json.loads(Path(plan['probes']).read_text())
    if not isinstance(probes,list) or not 1<=len(probes)<=plan['max_cases']:
        raise ValueError('nonempty bounded numerical probes required')
    seen=set()
    for probe in probes:
        ids=probe['input_ids']
        if not isinstance(probe.get('case_id'),str) or not probe['case_id'] or probe['case_id'] in seen:
            raise ValueError('unique numerical probe identities required')
        seen.add(probe['case_id'])
        if not isinstance(ids,list) or not 1<=len(ids)<=plan['max_input_tokens'] or any(type(token) is not int or not 0<=token<vocab for token in ids):
            raise ValueError('invalid or excessive fixed probe tokens')
        if len(ids)*vocab>plan['max_probe_logits_elements']:
            raise ValueError('probe forward-logit allocation exceeds budget')
    return merge_plan,merged,probes


def worker(plan_path,sha256,out):
    if digest(plan_path)!=sha256:
        raise ValueError('numerical plan changed before execution')
    plan=json.loads(Path(plan_path).read_text())
    base,merged,probes=validate(plan)
    check_packages(base)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    configure_memory(base)
    torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    attention_backend=plan.get('attention_backend','auto')
    if attention_backend=='math':
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(out)
    def snapshots(model,label):
        model.eval()
        results=[]
        for index,probe in enumerate(probes):
            ids=torch.tensor([probe['input_ids']],device=base['device'])
            with torch.inference_mode():
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits[0,-1].float().cpu()
                sequence=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,num_beams=1,max_new_tokens=plan['max_new_tokens'])[0].cpu().tolist()
            if sequence[:len(probe['input_ids'])]!=probe['input_ids'] or len(sequence)-len(probe['input_ids'])>plan['max_new_tokens']:
                raise ValueError('backend changed fixed prompt or exceeded generation cap')
            artifact=root/(label+'_'+str(index)+'.json')
            raw=dict(input_ids=probe['input_ids'],next_token_logits=logits.tolist(),generated_ids=sequence[len(probe['input_ids']):])
            torch.save(raw,root/(label+'_'+str(index)+'.pt'))
            write_json(artifact,raw)
            results.append(dict(case_id=probe['case_id'],logits=logits.tolist(),generated_ids=sequence[len(probe['input_ids']):],artifact=str(artifact),sha256=digest(artifact)))
        return results
    model=AutoModelForCausalLM.from_pretrained(base['base_model_path'],revision=base['base_revision'],local_files_only=True,trust_remote_code=False,dtype=getattr(torch,base['dtype']),attn_implementation='sdpa').to(base['device'])
    model=PeftModel.from_pretrained(model,base['adapter_path'],local_files_only=True,is_trainable=False)
    before=snapshots(model,'adapter')
    control=None
    if plan.get('negative_control')=='disabled_adapter':
        with model.disable_adapter():control=snapshots(model,'disabled_adapter')
    del model
    if base['device']=='cuda':
        torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(merged['model_path'],local_files_only=True,trust_remote_code=False,dtype=getattr(torch,base['dtype']),attn_implementation='sdpa').to(base['device'])
    after=snapshots(model,'merged')
    comparisons=[]
    for left,right in zip(before,after):
        comparison=compare_logits(left['logits'],right['logits'],plan['logit_atol'],plan.get('logit_rtol',0))
        comparisons.append(dict(case_id=left['case_id'],logits=comparison,generation_equal=left['generated_ids']==right['generated_ids']))
    negative=None
    if control is not None:
        checks=[dict(case_id=a['case_id'],logits=compare_logits(a['logits'],b['logits'],plan['logit_atol'],plan.get('logit_rtol',0)),generation_equal=a['generated_ids']==b['generated_ids']) for a,b in zip(before,control)]
        negative=dict(kind='disabled_adapter',detected=any(not c['logits']['within_tolerance'] for c in checks),comparisons=checks,records=[{k:v for k,v in r.items() if k!='logits'} for r in control])
    for row in before+after:row.pop('logits')
    rate=sum(row['generation_equal'] for row in comparisons)/len(comparisons)
    passed=all(row['logits']['within_tolerance'] for row in comparisons) and rate>=plan['min_generation_match_rate'] and (negative is None or negative['detected'])
    report=dict(attention_backend=attention_backend,sdpa_backends=dict(math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled()),model_attention_geometry=dict(query_heads=getattr(model.config,'num_attention_heads',None),kv_heads=getattr(model.config,'num_key_value_heads',None),head_dim=getattr(model.config,'head_dim',None)),tf32=False,negative_control=negative,numerical_report_version=NUMERIC_REPORT_VERSION,peak_allocated_bytes=torch.cuda.max_memory_allocated() if base['device']=='cuda' else None,passed=False,passed_numeric=passed,tool_behavior='pending',logit_positions='last_input_token_full_vocabulary',generation_match_rate=rate,
                thresholds=dict(logit_atol=plan['logit_atol'],logit_rtol=plan.get('logit_rtol',0),min_generation_match_rate=plan['min_generation_match_rate']),comparisons=comparisons,before=before,after=after,
                source_checkpoint_files_sha256=merged['training_source']['source_checkpoint_files_sha256'],optimizer_step=base['optimizer_step'],exported_checkpoint_sha256=merged['exported_checkpoint_sha256'],plan_sha256=sha256)
    replay_comparisons(report,probes,model.config.vocab_size,plan['logit_atol'],plan.get('logit_rtol',0))
    replay_control(report,probes,model.config.vocab_size,plan['logit_atol'],plan.get('logit_rtol',0),plan.get('negative_control'))
    validate(plan)  # Reject source/probe changes during the model computations.
    write_json(root/'numeric_equivalence.json',report)
    return report


def verify(plan_path,out,*,own_process_group=True):
    plan=json.loads(Path(plan_path).read_text())
    base,_,_=validate(plan);check_packages(base)
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    frozen=root/'resolved_numeric_plan.json';write_json(frozen,plan)
    command=[sys.executable,'-m','officegym.training.equivalence','--worker-plan',str(frozen),'--sha256',digest(frozen),'--out',str(root)]
    with (root/'numeric.stdout.log').open('w') as stdout,(root/'numeric.stderr.log').open('w') as stderr:
        status=execute_bounded(command,stdout,stderr,plan['max_wall_seconds'],own_process_group=own_process_group)
    status['full_equivalence_verified']=False
    write_json(root/'numeric_process.json',status)
    return status


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan');source.add_argument('--worker-plan')
    parser.add_argument('--sha256');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=worker(args.worker_plan,args.sha256,args.out) if args.worker_plan else verify(args.plan,args.out)
    print(json.dumps(result,indent=2))
    if result.get('exit_code',0)!=0 or result.get('wall_budget_exhausted') or result.get('passed_numeric') is False:
        raise SystemExit(1)

"""Bounded local fixed-token numerical checks; tool equivalence stays separate."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from officegym.eval.artifacts import write_json
from .merge import validate_merge_plan,check_packages
from .checkpoints import checkpoint_fingerprint
from .launch import execute_bounded


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_logits(left,right,atol):
    if not left or len(left)!=len(right):
        raise ValueError('unaligned nonempty probe logits')
    if type(atol) not in (int,float) or not math.isfinite(atol) or atol<0:
        raise ValueError('finite nonnegative logit tolerance required')
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in left+right):
        return dict(finite=False,max_abs_error=None,within_tolerance=False)
    error=max(abs(a-b) for a,b in zip(left,right))
    return dict(finite=True,max_abs_error=error,within_tolerance=error<=atol)


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
    torch.manual_seed(0);torch.use_deterministic_algorithms(True)
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
            artifact=root/(label+'_'+str(index)+'.pt')
            torch.save(dict(input_ids=probe['input_ids'],next_token_logits=logits,generated_ids=sequence[len(probe['input_ids']):]),artifact)
            results.append(dict(case_id=probe['case_id'],logits=logits.tolist(),generated_ids=sequence[len(probe['input_ids']):],artifact=str(artifact),sha256=digest(artifact)))
        return results
    model=AutoModelForCausalLM.from_pretrained(base['base_model_path'],revision=base['base_revision'],local_files_only=True,trust_remote_code=False,torch_dtype=getattr(torch,base['dtype'])).to(base['device'])
    model=PeftModel.from_pretrained(model,base['adapter_path'],local_files_only=True,is_trainable=False)
    before=snapshots(model,'adapter')
    del model
    if base['device']=='cuda':
        torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(merged['model_path'],local_files_only=True,trust_remote_code=False,torch_dtype=getattr(torch,base['dtype'])).to(base['device'])
    after=snapshots(model,'merged')
    comparisons=[]
    for left,right in zip(before,after):
        comparison=compare_logits(left.pop('logits'),right.pop('logits'),plan['logit_atol'])
        comparisons.append(dict(case_id=left['case_id'],logits=comparison,generation_equal=left['generated_ids']==right['generated_ids']))
    rate=sum(row['generation_equal'] for row in comparisons)/len(comparisons)
    passed=all(row['logits']['within_tolerance'] for row in comparisons) and rate>=plan['min_generation_match_rate']
    report=dict(passed=False,passed_numeric=passed,tool_behavior='pending',logit_positions='last_input_token_full_vocabulary',generation_match_rate=rate,
                thresholds=dict(logit_atol=plan['logit_atol'],min_generation_match_rate=plan['min_generation_match_rate']),comparisons=comparisons,before=before,after=after,
                source_checkpoint_files_sha256=merged['training_source']['source_checkpoint_files_sha256'],optimizer_step=base['optimizer_step'],exported_checkpoint_sha256=merged['exported_checkpoint_sha256'],plan_sha256=sha256)
    validate(plan)  # Reject source/probe changes during the model computations.
    write_json(root/'numeric_equivalence.json',report)
    return report


def verify(plan_path,out):
    plan=json.loads(Path(plan_path).read_text())
    base,_,_=validate(plan);check_packages(base)
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    frozen=root/'resolved_numeric_plan.json';write_json(frozen,plan)
    command=[sys.executable,'-m','officegym.training.equivalence','--worker-plan',str(frozen),'--sha256',digest(frozen),'--out',str(root)]
    with (root/'numeric.stdout.log').open('w') as stdout,(root/'numeric.stderr.log').open('w') as stderr:
        status=execute_bounded(command,stdout,stderr,plan['max_wall_seconds'])
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

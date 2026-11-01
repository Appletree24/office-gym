"""Explicit sixteen-sample teacher-forced overfit measurements, not task success."""
import hashlib
import math
import time
from officegym.env.world import canonical


def validate_config(config):
    if not isinstance(config,dict):raise ValueError('explicit overfit criteria required')
    for key in ('max_mean_nll','min_token_accuracy','min_relative_nll_reduction','max_evaluation_wall_seconds'):
        value=config.get(key)
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite predeclared overfit criteria required: '+key)
    if config['max_mean_nll']<=0 or config['max_evaluation_wall_seconds']<=0 or any(not 0<=config[key]<=1 for key in ('min_token_accuracy','min_relative_nll_reduction')):raise ValueError('invalid overfit criterion bounds')
    return config


def validate_rows(rows):
    from .sft import validate_samples
    validate_samples(rows,8192)
    if len(rows)!=16:raise ValueError('overfit check requires exactly sixteen samples')
    ids=[row.get('instance_id') for row in rows]
    signatures=[canonical([row.get('instance_id'),row['input_ids'],row['labels']]) for row in rows]
    if any(not isinstance(value,str) or not value for value in ids) or len(set(signatures))!=16:raise ValueError('sixteen distinct source samples required')
    if any(not any(label!=-100 for label in row['labels'][1:]) for row in rows):raise ValueError('each overfit sample needs causally predicted assistant tokens')


def measure(model,rows,*,deadline):
    """Evaluate original shifted labels in bounded chunks; preserve train/eval mode."""
    validate_rows(rows)
    if type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('finite evaluation deadline required')
    import torch
    if time.monotonic()>=deadline:raise TimeoutError('overfit evaluation wall budget exhausted')
    training=model.training;model.eval();records=[];device=next(model.parameters()).device
    try:
        with torch.no_grad():
            for row in rows:
                if time.monotonic()>=deadline:raise TimeoutError('overfit evaluation wall budget exhausted')
                ids=torch.tensor([row['input_ids']],dtype=torch.long,device=device)
                labels=torch.tensor(row['labels'][1:],dtype=torch.long,device=device)
                positions=torch.nonzero(labels!=-100,as_tuple=False).flatten()
                outputs=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
                logits=outputs.logits
                if logits.ndim!=3 or tuple(logits.shape[:2])!=tuple(ids.shape):raise ValueError('unaligned causal model logits')
                total=0.;correct=0
                for start in range(0,len(positions),128):
                    if time.monotonic()>=deadline:raise TimeoutError('overfit evaluation wall budget exhausted')
                    selected=positions[start:start+128];scores=logits[0,selected,:].float();targets=labels[selected]
                    losses=torch.nn.functional.cross_entropy(scores,targets,reduction='none')
                    if not torch.isfinite(losses).all():raise ValueError('nonfinite assistant overfit loss')
                    total+=losses.double().sum().item();correct+=(scores.argmax(dim=-1)==targets).sum().item()
                    del scores,targets,losses
                count=len(positions)
                records.append(dict(instance_id=row['instance_id'],supervised_tokens=count,nll_sum=total,correct_tokens=correct,sequence_exact=correct==count))
                del outputs,logits,ids,labels,positions
        if time.monotonic()>=deadline:raise TimeoutError('overfit evaluation wall budget exhausted')
    finally:model.train(training)
    tokens=sum(row['supervised_tokens'] for row in records);correct=sum(row['correct_tokens'] for row in records)
    return dict(schema='officegym-sixteen-sample-measurement-v1',samples=16,rows_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),supervised_tokens=tokens,
                mean_nll=sum(row['nll_sum'] for row in records)/tokens,token_accuracy=correct/tokens,sequence_exact_accuracy=sum(row['sequence_exact'] for row in records)/16,per_sample=records,
                scope='Teacher-forced assistant tokens only; not generated tool validity or environment success.')


def compare(before,after,criteria):
    validate_config(criteria)
    for record in (before,after):
        if record.get('schema')!='officegym-sixteen-sample-measurement-v1' or record.get('samples')!=16:raise ValueError('sixteen-sample measurements required')
        if type(record.get('supervised_tokens')) is not int or record['supervised_tokens']<16:raise ValueError('positive per-sample supervised token evidence required')
        for key in ('mean_nll','token_accuracy','sequence_exact_accuracy'):
            if type(record.get(key)) not in (int,float) or not math.isfinite(record[key]) or record[key]<0:raise ValueError('finite nonnegative overfit metrics required')
        if record['token_accuracy']>1 or record['sequence_exact_accuracy']>1:raise ValueError('invalid accuracy range')
    if before['rows_sha256']!=after['rows_sha256'] or before['supervised_tokens']!=after['supervised_tokens']:raise ValueError('overfit measurement data changed')
    reduction=(before['mean_nll']-after['mean_nll'])/max(before['mean_nll'],1e-12)
    checks=dict(mean_nll=after['mean_nll']<=criteria['max_mean_nll'],token_accuracy=after['token_accuracy']>=criteria['min_token_accuracy'],relative_nll_reduction=reduction>=criteria['min_relative_nll_reduction'])
    return dict(schema='officegym-sixteen-sample-overfit-v1',passed=all(checks.values()),checks=checks,criteria=criteria,relative_nll_reduction=reduction,before=before,after=after,independent_dev_evaluation_status='pending',environment_success_measured=False)

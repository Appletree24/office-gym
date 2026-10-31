"""Content-addressed checkpoint evidence and SFT branch invariants."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical


def checkpoint_fingerprint(path):
    root = Path(path).resolve()
    if not root.is_dir() or not (root/'config.json').is_file():
        raise ValueError('checkpoint directory/config missing')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.bin','.model','.jinja','.txt'))
    if not any(p.suffix in ('.safetensors','.bin') for p in files):
        raise ValueError('checkpoint has no weight files')
    hashes = {}
    for file in files:
        if file.is_symlink() and not file.resolve().is_relative_to(root):
            raise ValueError('checkpoint contains external symlink; materialize export first')
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(chunk)
        hashes[str(file.relative_to(root))] = digest.hexdigest()
    return {'checkpoint_sha256':hashlib.sha256(canonical(hashes).encode()).hexdigest(),'files_sha256':hashes}


def record_export(checkpoint,out,*,stage,parent_hash,training_manifest_hash,versions,equivalence_report,training_events=None,source_checkpoint=None,optimizer_step=None):
    if stage not in ('base','sft','grpo','rs_sft'):
        raise ValueError('unknown checkpoint stage')
    if stage!='base' and (not parent_hash or not training_manifest_hash):
        raise ValueError('training checkpoint lineage missing')
    source=None
    if stage!='base':
        if training_events is None or source_checkpoint is None:raise ValueError('saved training checkpoint evidence required')
        source=bind_formal_training_source(training_events,source_checkpoint,optimizer_step,stage,parent_hash,training_manifest_hash)
    if not equivalence_report or not Path(equivalence_report).is_file():
        raise ValueError('export equivalence report required')
    report_path = Path(equivalence_report)
    report = json.loads(report_path.read_text())
    if report.get('passed') is not True or not report.get('checks'):
        raise ValueError('export equivalence gate not passed')
    fingerprint=checkpoint_fingerprint(checkpoint)
    if report.get('exported_checkpoint_sha256')!=fingerprint['checkpoint_sha256']:
        raise ValueError('equivalence report does not bind exported checkpoint contents')
    if stage!='base':
        if report.get('source_checkpoint_files_sha256')!=source['source_checkpoint_files_sha256'] or report.get('optimizer_step')!=optimizer_step:
            raise ValueError('equivalence report differs from training checkpoint source')
    record = dict(fingerprint,checkpoint_path=str(Path(checkpoint).resolve()),export_record_version='checkpoint-export-v3-formal-admission',training_source=source,stage=stage,parent_hash=parent_hash,training_manifest_hash=training_manifest_hash,versions=versions,
                  equivalence_report=str(report_path.resolve()),equivalence_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    with Path(out).open('x') as stream:
        stream.write(canonical(record)+'\n')
    return record


def validate_branches(sft,grpo_start,reference,rs_start):
    if sft.get('stage')!='sft':
        raise ValueError('branch point must be exported SFT checkpoint')
    for record in (sft,grpo_start,reference,rs_start):
        if record.get('stage')!='sft':raise ValueError('all branch starts must be admitted SFT exports')
        verify_export_admission(record)
    target = sft['checkpoint_sha256']
    if any(record['checkpoint_sha256']!=target for record in (grpo_start,reference,rs_start)):
        raise ValueError('GRPO actor/reference and RS-SFT must share identical SFT checkpoint')
    return target


def bind_training_source(events_path,checkpoint,optimizer_step):
    from .evidence import checkpoint_inventory
    if type(optimizer_step) is not int or optimizer_step<0:
        raise ValueError('recorded optimizer step required')
    checkpoint=Path(checkpoint).resolve()
    inventory=checkpoint_inventory(checkpoint)
    prefix=hashlib.sha256();length=0
    with Path(events_path).open('rb') as stream:
        for line in stream:
            prefix.update(line);length+=len(line)
            row=json.loads(line)
            if row.get('kind') not in ('training_checkpoint','final_adapter') or row.get('optimizer_step')!=optimizer_step:
                continue
            payload=row.get('payload',{})
            if payload.get('path')!=str(checkpoint):
                continue
            if payload!=inventory:
                raise ValueError('source checkpoint changed since its recorded training step')
            return dict(events_path=str(Path(events_path).resolve()),prefix_bytes=length,prefix_sha256=prefix.hexdigest(),
                        optimizer_step=optimizer_step,source_checkpoint=str(checkpoint),source_checkpoint_files_sha256=inventory['checkpoint_files_sha256'])
    raise ValueError('no saved checkpoint event at the claimed optimizer step')


def bind_formal_training_source(events_path,checkpoint,optimizer_step,stage,parent_hash,manifest_hash):
    from .admission import validate_admission
    if type(optimizer_step) is not int or optimizer_step<1:raise ValueError('formal export requires positive optimizer step')
    with Path(events_path).open('rb') as stream:
        line=stream.readline()
    header=json.loads(line) if line else {}
    if header.get('kind')!='training_admission' or header.get('optimizer_step')!=0:
        raise ValueError('formal training admission missing; backend probes cannot become formal exports')
    payload=validate_admission(header.get('payload',{}))
    if optimizer_step>payload['max_optimizer_steps']:raise ValueError('claimed optimizer step exceeds admitted budget')
    if payload['stage']!=stage or payload['parent_checkpoint_sha256']!=parent_hash or payload['data_manifest']['sha256']!=manifest_hash:
        raise ValueError('export lineage differs from admitted training stage/parent/data')
    source=bind_training_source(events_path,checkpoint,optimizer_step)
    if stage=='grpo':
        verify_grpo_source_lock(events_path,source['prefix_bytes'])
        plan=json.loads(Path(payload['config']['path']).read_text())
        from .verl_entry import load_resolved
        resolved=load_resolved(plan['resolved_upstream_config'],plan['resolved_upstream_sha256'])
        expected_rows=resolved['data']['train_batch_size']*resolved['actor_rollout_ref']['rollout']['n']
        verify_grpo_updates(events_path,source['prefix_bytes'],optimizer_step,initial_actor=parent_hash,expected_rows=expected_rows)
    if not Path(checkpoint).resolve().is_relative_to(Path(events_path).resolve().parent):raise ValueError('formal checkpoint escapes admitted training run')
    return dict(source,admission=payload)


def verify_grpo_source_lock(events_path,prefix_bytes):
    from .verl_checkpoint import SOURCE_HASHES
    records=list(map(json.loads,Path(events_path).read_bytes()[:prefix_bytes].splitlines()))
    locks=[row for row in records if row.get('kind')=='backend_source_lock']
    if len(locks)!=1 or locks[0].get('optimizer_step')!=0 or locks[0].get('payload')!=SOURCE_HASHES:
        raise ValueError('GRPO source lacks the pinned native controller source lock')
    return dict(SOURCE_HASHES)


def verify_grpo_updates(events_path,prefix_bytes,optimizer_step,*,initial_actor=None,expected_rows=None):
    from math import isfinite
    from .actor_version import actor_identity
    raw=Path(events_path).read_bytes()
    if type(prefix_bytes) is not int or not 0<prefix_bytes<=len(raw) or type(optimizer_step) is not int or optimizer_step<1:
        raise ValueError('bounded positive GRPO evidence prefix/step required')
    records=list(map(json.loads,raw[:prefix_bytes].splitlines()))
    updates=[row for row in records if row.get('kind')=='optimizer_update']
    if [r['optimizer_step'] for r in updates]!=list(range(1,optimizer_step+1)):raise ValueError('GRPO source lacks complete optimizer-update evidence')
    if initial_actor is None:initial_actor=updates[0].get('payload',{}).get('sampled_actor_version')
    if not isinstance(initial_actor,str) or not initial_actor:raise ValueError('admitted initial actor identity required')
    fields=('selected_trajectories','actor_loss_tokens','forward_context_tokens','selected_generated_tokens','context_only_tokens')
    pending=None;completed=0
    def usage(proof):
        if any(type(proof.get(key)) is not int or proof[key]<0 for key in fields):raise ValueError('GRPO update lacks integer training-token evidence')
        if proof['selected_trajectories']<1 or proof['actor_loss_tokens']<1 or proof['selected_generated_tokens']!=proof['actor_loss_tokens'] or proof['forward_context_tokens']!=proof['actor_loss_tokens']+proof['context_only_tokens']:
            raise ValueError('inconsistent GRPO training-token accounting')
        if expected_rows is not None and proof['selected_trajectories']!=expected_rows:raise ValueError('GRPO selected rows differ from admission')
        return tuple(proof[key] for key in fields)
    for row in records:
        if row.get('kind') not in ('optimizer_update_attempt','optimizer_update'):continue
        step=row.get('optimizer_step');proof=row.get('payload',{})
        if type(step) is not int or step!=completed+1 or proof.get('native_global_step')!=step or type(proof.get('native_global_step')) is not int:
            raise ValueError('GRPO update attempt/order differs from verified prefix')
        counts=usage(proof)
        if row['kind']=='optimizer_update_attempt':
            if pending is not None:raise ValueError('duplicate or unresolved GRPO optimizer attempt')
            pending=counts;continue
        if pending is None or counts!=pending:raise ValueError('GRPO update lacks its matching attempted batch')
        norms=proof.get('gradient_norms')
        if proof.get('contract')!='single-mini-batch-epoch' or not isinstance(norms,list) or not norms or any(type(v) not in (int,float) or not isfinite(v) or v<0 for v in norms):raise ValueError('invalid GRPO optimizer-update evidence')
        metrics=proof.get('native_actor_metrics',{});kl=metrics.get('actor/kl_loss') if isinstance(metrics,dict) else None
        if type(kl) not in (int,float) or not isfinite(kl):raise ValueError('GRPO update lacks finite native reference KL')
        if type(proof.get('sampled_actor_step')) is not int or proof['sampled_actor_step']!=step-1 or proof.get('sampled_actor_version')!=actor_identity(initial_actor,step-1) or proof.get('per_generation_version_checks') is not True:
            raise ValueError('GRPO update lacks admitted sampled actor identity')
        pending=None;completed+=1
    if pending is not None or completed!=optimizer_step:raise ValueError('unresolved optimizer attempt before source checkpoint')
    return optimizer_step


def verify_export_admission(record):
    from .admission import digest
    if record.get('export_record_version')!='checkpoint-export-v3-formal-admission' or not record.get('training_source',{}).get('admission'):
        raise ValueError('formal export admission provenance required')
    if checkpoint_fingerprint(record['checkpoint_path'])['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('formal exported checkpoint changed')
    source=record['training_source']
    current=bind_formal_training_source(source['events_path'],source['source_checkpoint'],source['optimizer_step'],record['stage'],record['parent_hash'],record['training_manifest_hash'])
    if current!=source:raise ValueError('formal export training admission changed')
    path=Path(record['equivalence_report'])
    if not path.is_absolute() or digest(path)!=record['equivalence_report_sha256']:raise ValueError('formal export equivalence evidence changed')
    report=json.loads(path.read_text())
    if report.get('passed') is not True or report.get('exported_checkpoint_sha256')!=record['checkpoint_sha256'] or report.get('source_checkpoint_files_sha256')!=source['source_checkpoint_files_sha256'] or report.get('optimizer_step')!=source['optimizer_step']:
        raise ValueError('formal export equivalence identity changed')
    return source['optimizer_step']


def verify_training_source(source):
    if source.get('admission'):
        admitted=source['admission']
        current=bind_formal_training_source(source['events_path'],source['source_checkpoint'],source['optimizer_step'],admitted['stage'],admitted['parent_checkpoint_sha256'],admitted['data_manifest']['sha256'])
    else:current=bind_training_source(source['events_path'],source['source_checkpoint'],source['optimizer_step'])
    if current!=source:
        raise ValueError('checkpoint training evidence changed')
    return current['optimizer_step']

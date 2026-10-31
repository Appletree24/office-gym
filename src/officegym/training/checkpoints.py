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
    if not equivalence_report or not Path(equivalence_report).is_file():
        raise ValueError('export equivalence report required')
    report_path = Path(equivalence_report)
    report = json.loads(report_path.read_text())
    if report.get('passed') is not True or not report.get('checks'):
        raise ValueError('export equivalence gate not passed')
    fingerprint=checkpoint_fingerprint(checkpoint)
    if report.get('exported_checkpoint_sha256')!=fingerprint['checkpoint_sha256']:
        raise ValueError('equivalence report does not bind exported checkpoint contents')
    source=None
    if stage!='base':
        if training_events is None or source_checkpoint is None:
            raise ValueError('saved training checkpoint evidence required')
        source=bind_training_source(training_events,source_checkpoint,optimizer_step)
        if report.get('source_checkpoint_files_sha256')!=source['source_checkpoint_files_sha256'] or report.get('optimizer_step')!=optimizer_step:
            raise ValueError('equivalence report differs from training checkpoint source')
    record = dict(fingerprint,export_record_version='checkpoint-export-v2-training-source',training_source=source,stage=stage,parent_hash=parent_hash,training_manifest_hash=training_manifest_hash,versions=versions,
                  equivalence_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    with Path(out).open('x') as stream:
        stream.write(canonical(record)+'\n')
    return record


def validate_branches(sft,grpo_start,reference,rs_start):
    if sft.get('stage')!='sft':
        raise ValueError('branch point must be exported SFT checkpoint')
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


def verify_training_source(source):
    current=bind_training_source(source['events_path'],source['source_checkpoint'],source['optimizer_step'])
    if current!=source:
        raise ValueError('checkpoint training evidence changed')
    return current['optimizer_step']

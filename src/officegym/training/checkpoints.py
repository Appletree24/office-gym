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


def record_export(checkpoint,out,*,stage,parent_hash,training_manifest_hash,versions,equivalence_report):
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
    record = dict(checkpoint_fingerprint(checkpoint),stage=stage,parent_hash=parent_hash,training_manifest_hash=training_manifest_hash,versions=versions,
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

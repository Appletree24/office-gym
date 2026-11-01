"""Revalidate retained teacher evidence before SFT admission, without model calls."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from .data import verify_mask
from .teacher import extract_trajectory

FILES=('result.json','trajectory.jsonl','policy_isolation.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(path):return {name:digest(Path(path)/name) for name in FILES}


def verify_sources(data_dir,manifest):
    source=Path(data_dir)
    rows=[json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines() if line.strip()]
    conversations=[json.loads(line) for line in (source/'conversations.jsonl').read_text().splitlines() if line.strip()]
    accepted=manifest['accepted']
    if not rows or len(rows)!=len(conversations) or len(rows)!=len(accepted):raise ValueError('SFT source/sample/conversation counts differ')
    paths=set();signatures=set();tokens=0
    for row,conversation,entry in zip(rows,conversations,accepted):
        path=Path(entry.get('source_run',''))
        if not path.is_absolute() or str(path)!=row.get('source_run') or path.resolve() in paths:raise ValueError('unique absolute SFT teacher source paths required')
        paths.add(path.resolve())
        expected=entry.get('source_files_sha256')
        if not isinstance(expected,dict) or set(expected)!=set(FILES):raise ValueError('complete hash-bound teacher evidence required')
        if source_hashes(path)!=expected:raise ValueError('retained teacher evidence changed')
        recovered=extract_trajectory(path)
        if recovered!=conversation:raise ValueError('SFT conversation differs from retained teacher trace')
        identity=hashlib.sha256(canonical(recovered).encode()).hexdigest()
        if row.get('trajectory_sha256')!=identity or entry.get('trajectory_sha256')!=identity or row['instance_id']!=recovered['instance_id'] or entry.get('instance_id')!=row['instance_id']:raise ValueError('SFT sample source identity differs')
        verify_mask(row['input_ids'],row['assistant_mask'],row['token_roles'])
        if row['labels']!=[token if mask else -100 for token,mask in zip(row['input_ids'],row['assistant_mask'])]:raise ValueError('SFT source labels differ from assistant mask')
        count=len(row['input_ids'])
        if count!=entry.get('tokens') or count>manifest['max_length'] or count>8192:raise ValueError('SFT source token length differs or exceeds training cap')
        signature=hashlib.sha256(canonical(row['input_ids']).encode()).hexdigest()
        if signature in signatures:raise ValueError('duplicate SFT token sequence')
        signatures.add(signature);tokens+=count
    if tokens!=manifest.get('training_tokens'):raise ValueError('SFT source training token total differs')
    return dict(samples=len(rows),training_tokens=tokens,source_files=len(FILES)*len(rows),scope='Retained-run provenance revalidation; not independent provider identity authentication.')

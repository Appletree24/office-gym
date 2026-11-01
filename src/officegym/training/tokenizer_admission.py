"""Check the actual locally loaded training tokenizer against every saved sample."""
import hashlib
import json
import math
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.harness.tokenizer import PromptTokenizer
from .tokenizer import SFTTokenizer


def validate_runtime(plan,manifest,data_dir,rows,tokenizer,*,deadline):
    if type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('finite tokenizer validation deadline required')
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('training wall budget exhausted during tokenizer verification')
    check_time()
    revision=plan.get('tokenizer_revision',plan['model_revision'])
    prompt=PromptTokenizer(tokenizer,model_revision=revision,template_kwargs=plan.get('template_kwargs'))
    actual=prompt.manifest()
    if actual!=manifest.get('tokenizer'):raise ValueError('runtime tokenizer differs from dataset revision/vocabulary/template/kwargs')
    native=SFTTokenizer(tokenizer,template_kwargs=plan.get('template_kwargs'))
    if native.mask_manifest()!=manifest.get('mask_tokenizer'):raise ValueError('runtime assistant-mask template differs from dataset')
    source=Path(data_dir);payload=(source/'samples.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['samples_sha256'] or [json.loads(line) for line in payload.splitlines() if line.strip()]!=rows:raise ValueError('runtime sample materialization differs from admitted data')
    conversation_payload=(source/'conversations.jsonl').read_bytes()
    conversations=[json.loads(line) for line in conversation_payload.splitlines() if line.strip()]
    if not rows or len(rows)!=len(conversations) or len(rows)!=len(manifest['accepted']):raise ValueError('runtime tokenizer source/sample counts differ')
    total=supervised=0
    for row,conversation in zip(rows,conversations):
        check_time()
        if row['instance_id']!=conversation['instance_id'] or row['trajectory_sha256']!=hashlib.sha256(canonical(conversation).encode()).hexdigest():raise ValueError('runtime source conversation differs from sample')
        encoded=native(conversation['messages'],conversation['tools'])
        if any(encoded[key]!=row[key] for key in ('input_ids','assistant_mask','token_roles')):raise ValueError('native runtime token/role/mask sequence differs from saved sample')
        if row['labels']!=[token if mask else -100 for token,mask in zip(row['input_ids'],row['assistant_mask'])]:raise ValueError('runtime labels differ from native mask')
        if not any(row['assistant_mask'][1:]):raise ValueError('runtime sample has no causally supervised targets')
        total+=len(row['input_ids']);supervised+=sum(row['assistant_mask'][1:])
    check_time()
    return dict(schema='officegym-runtime-tokenizer-check-v1',tokenizer=actual,mask_tokenizer=native.mask_manifest(),samples=len(rows),tokens=total,causally_supervised_tokens=supervised,samples_sha256=manifest['samples_sha256'],conversations_sha256=hashlib.sha256(conversation_payload).hexdigest(),all_native_sequences_equal=True,
                scope='Exact saved-corpus tokenizer equivalence; no model optimization, generation or task-success measurement.')

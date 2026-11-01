"""Retain and branch a quiescent committed-but-unacknowledged invitation boundary."""
import hashlib
import json
import sqlite3
from pathlib import Path
from officegym.env.rollout import clone_world
from officegym.env.world import canonical
from .plan import freeze_plan

FILES=('world.sqlite','runtime.sqlite','harness.jsonl')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary(root):
    root=Path(root)
    world=sqlite3.connect((root/'world.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    runtime=sqlite3.connect((root/'runtime.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        world.row_factory=runtime.row_factory=sqlite3.Row
        pending=runtime.execute("SELECT * FROM write_intents WHERE status='PREPARED'").fetchall()
        if len(pending)!=1:raise ValueError('exactly one unacknowledged invitation required')
        intent=pending[0];args=json.loads(intent['args_json']);operation=intent['operation_id']
        if intent['tool_name']!='mail.send' or args.get('kind')!='invitation' or intent['result_json'] is not None:
            raise ValueError('pending action is not an unacknowledged invitation')
        principal=json.loads(world.execute("SELECT value_json FROM meta WHERE key='principal_id'").fetchone()[0])
        actual=world.execute('SELECT * FROM operations WHERE operation_id=?',(operation,)).fetchone()
        if actual is None or actual['tool_name']!='mail.send' or actual['principal_id']!=principal:
            raise ValueError('pending invitation has no committed world operation')
        if intent['request_sha256']!=hashlib.sha256(canonical(['mail.send',args]).encode()).hexdigest() or actual['request_sha256']!=hashlib.sha256(canonical(['mail.send',args,principal]).encode()).hexdigest():
            raise ValueError('pending and committed invitation requests differ')
        receipt=json.loads(actual['response_json'])
        effects=world.execute('SELECT * FROM outbox WHERE operation_id=? ORDER BY recipient_id',(operation,)).fetchall()
        if sorted(r['recipient_id'] for r in effects)!=sorted(args['recipient_ids']) or sorted(receipt['recipient_ids'])!=sorted(args['recipient_ids']) or any(r['message_id']!=receipt['message_id'] or r['kind']!='invitation' or r['meeting_id']!=args['meeting_id'] or r['meeting_version']!=args['meeting_version'] for r in effects):
            raise ValueError('committed invitation effects differ from original request')
        records=[json.loads(line) for line in (root/'harness.jsonl').read_text().splitlines() if line.strip()]
        if not records or records[0].get('kind')!='run_header' or records[0].get('mode')!='full':raise ValueError('original full Harness header required')
        actions=[row for row in records if row.get('kind')=='action_start' and row.get('operation_id')==operation]
        if len(actions)!=1 or actions[0]['tool']!='mail.send' or actions[0]['arguments']!=args or any(r.get('kind')=='interaction' and r.get('action_id')==actions[0]['action_id'] for r in records):
            raise ValueError('trace is not at the unacknowledged commit boundary')
        return dict(operation_id=operation,action_id=actions[0]['action_id'],recipient_ids=sorted(args['recipient_ids']),meeting_id=args['meeting_id'],meeting_version=args['meeting_version'],outbox_effects=len(effects),model_version=records[0].get('model_version'),harness_version=records[0]['harness_version'],task_sha256=records[0]['task_sha256'],budget=records[0]['budget'])
    finally:world.close();runtime.close()


def capture_checkpoint(world,runtime,trace,out):
    """Caller must have stopped its worker and quiesced further world requests."""
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    for source,name in ((world,'world.sqlite'),(runtime,'runtime.sqlite')):
        snapshot=clone_world(source,root/name);snapshot.close()
    with (root/'harness.jsonl').open('xb') as stream:stream.write(Path(trace).read_bytes())
    record=dict(schema='officegym-fault-checkpoint-v1',protocol='after_world_commit',boundary=boundary(root),files_sha256={name:digest(root/name) for name in FILES},scope='Content-bound boundary snapshot; caller owns worker-kill/quiescence evidence and model provenance. Not formal fault evaluation.')
    freeze_plan([record],root/'checkpoint.json')
    return record


def verify_checkpoint(root):
    root=Path(root);path=root/'checkpoint.json'
    if Path(str(path)+'.sha256').read_text().strip()!=digest(path):raise ValueError('fault checkpoint checksum mismatch')
    record=json.loads(path.read_text())
    if record.get('schema')!='officegym-fault-checkpoint-v1' or record.get('protocol')!='after_world_commit' or record.get('files_sha256')!={name:digest(root/name) for name in FILES}:
        raise ValueError('fault checkpoint contents changed')
    if record.get('boundary')!=boundary(root):raise ValueError('fault checkpoint boundary changed')
    return record


def branch_checkpoint(source,out):
    source=Path(source).resolve();record=verify_checkpoint(source);root=Path(out);root.mkdir(parents=True,exist_ok=False)
    for name in ('world.sqlite','runtime.sqlite'):
        world=clone_world(source/name,root/name);world.close()
    with (root/'harness.jsonl').open('xb') as stream:stream.write((source/'harness.jsonl').read_bytes())
    if boundary(root)!=record['boundary']:raise ValueError('cloned fault boundary differs')
    if verify_checkpoint(source)!=record:raise ValueError('source fault checkpoint changed while branching')
    binding=dict(source_checkpoint=str(source),source_checkpoint_sha256=digest(source/'checkpoint.json'),boundary=record['boundary'],scope='Independent database clone at the same commit boundary; no recovery mode or outcome inferred.')
    with (root/'branch.json').open('x') as stream:stream.write(canonical(binding)+'\n')
    return binding

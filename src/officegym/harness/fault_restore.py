"""Prespecified simple/full reconstruction of one shared committed fault prefix."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.eval.fault_checkpoint import verify_branch,digest
from officegym.tools.schema import tool_schemas
from .loop import initial_messages,wall_clock_identity
from .versions import HARNESS_VERSION

PROTOCOL='shared-fault-prefix-recovery-v1'


def restore_fault_checkpoint(harness,checkpoint):
    source=Path(checkpoint).resolve();root=Path(harness.trace_path).resolve().parent;trace=Path(harness.trace_path)
    if trace.exists():raise ValueError('fault branch requires a fresh continuation trace')
    if harness.restored:raise ValueError('fault branch context already restored')
    binding=verify_branch(root,source)
    if harness.journal is not None:
        journal_path=Path(harness.journal.db.execute('PRAGMA database_list').fetchone()[2]).resolve()
        if journal_path!=(root/'runtime.sqlite'):raise ValueError('full recovery must use its cloned branch journal')
    source_trace=root/'harness.jsonl'
    header=json.loads(source_trace.read_text().splitlines()[0])
    if header.get('tool_schemas')!=tool_schemas() or header.get('initial_messages')!=initial_messages(harness.task,full=True):
        raise ValueError('shared recovery source prompt/schema differs from frozen full prefix')
    prior=harness._restore_from_trace(source_trace,'full')
    pending=copy.deepcopy(harness.pending_actions)
    mode='full' if harness.journal is not None else 'simple'
    if mode=='simple':
        # Logged operation IDs are available to both methods. This is control
        # feedback, never a fabricated tool receipt or automatic reconciliation.
        notice=dict(kind='controller_recovery_notice',message='Execution stopped before observing the result of these logged actions. Their outcomes are unknown to the agent. Check public tool evidence before choosing further actions.',pending_actions=[{key:action.get(key) for key in ('tool','arguments','operation_id','content')} for action in pending.values()])
        harness.messages.append(dict(role='user',content=canonical(notice)))
        harness.pending_actions={}
    metadata=dict(protocol=PROTOCOL,recovery_mode=mode,source_checkpoint_sha256=binding['source_checkpoint_sha256'],source_trace_sha256=digest(source_trace),prior_usage=copy.deepcopy(prior),inherited_pending_actions=pending,clock_reset=False)
    records=[dict(kind='run_header',fault_branch=True,wall_clock=dict(monotonic_origin=harness.wall_origin,identity=wall_clock_identity()),harness_version=HARNESS_VERSION,task_sha256=hashlib.sha256(canonical(harness.task).encode()).hexdigest(),budget=asdict(harness.budget),mode=mode,model_version=getattr(harness.policy,'model_version',None),tool_choice=getattr(harness.policy,'tool_choice',None),initial_messages=initial_messages(harness.task,full=mode=='full'),tool_schemas=tool_schemas(),recovery=metadata),dict(kind='fault_context_restored',protocol=PROTOCOL,messages=copy.deepcopy(harness.messages),elapsed_wall_seconds=time.monotonic()-harness.wall_origin)]
    with trace.open('x') as stream:
        for row in records:stream.write(canonical(row)+'\n')
        stream.flush();os.fsync(stream.fileno())
    return metadata

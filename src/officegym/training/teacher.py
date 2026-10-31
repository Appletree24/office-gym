"""Recover SFT conversational source from a measured teacher episode."""
from officegym.graders.versions import require_current_scoring
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.harness.policy import render_messages
from .data import RejectedTrajectory


def extract_trajectory(run_dir):
    root = Path(run_dir)
    result = json.loads((root/'result.json').read_text())
    if result['split']!='train':
        raise RejectedTrajectory('non_training_split')
    if result.get('policy_kind')!='inference_service':
        raise RejectedTrajectory('not_a_model_trajectory')
    try:
        require_current_scoring(result.get('grade',{}))
    except ValueError as exc:
        raise RejectedTrajectory('stale_scoring_version') from exc
    if not result['grade']['full_success'] or result['termination']['state']!='DECLARED_DONE' or not result['operation_audit']['consistent']:
        raise RejectedTrajectory('not_verified_success')
    trace_path = root/'trajectory.jsonl'
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if not records or records[0].get('kind')!='run_header' or 'initial_messages' not in records[0] or 'tool_schemas' not in records[0]:
        raise RejectedTrajectory('missing_original_prompt')
    header = records[0]
    if header['mode']!='full' or header['model_version']!=result['model_version']:
        raise RejectedTrajectory('teacher_mode_or_model_mismatch')
    if any(row['kind']=='context_selection' and (row.get('compressed') or row.get('summary_output_tokens',0) or row.get('removed_groups',0)) for row in records):
        raise RejectedTrajectory('compressed_or_truncated')
    if any(row.get('reconciled') or row.get('retry_same_key') for row in records):
        raise RejectedTrajectory('recovered')
    messages = list(header['initial_messages'])
    pending = set()
    for row in records[1:]:
        if row['kind']=='action_start':
            pending.add(row['action_id'])
        elif row['kind']=='interaction':
            if row.get('action_id') not in pending:
                raise RejectedTrajectory('unmatched_interaction')
            pending.remove(row['action_id'])
            pair = row['messages']
            if len(pair)!=2 or pair[0].get('tool_call',{}).get('id')!=pair[1].get('tool_call_id'):
                raise RejectedTrajectory('broken_tool_pair')
            messages.extend(pair)
        elif row['kind']=='parse_failure' and row.get('message'):
            messages.append(row['message'])
    if pending:
        raise RejectedTrajectory('unresolved_operations')
    trajectory = dict(instance_id=result['instance_id'],split='train',grade=result['grade'],messages=messages,tools=header['tool_schemas'],
                      model_version=result['model_version'],compacted=False,recovered=False,source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
    trajectory['conversation'] = render_messages(messages)
    trajectory['content_sha256'] = hashlib.sha256(canonical(trajectory).encode()).hexdigest()
    return trajectory

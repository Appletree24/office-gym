"""Recover SFT conversational source from a measured teacher episode."""
from officegym.graders.versions import require_current_scoring
from officegym.harness.versions import HARNESS_VERSION
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
    if result.get('fault') is not None:
        raise RejectedTrajectory('fault_evaluation_source')
    try:
        require_current_scoring(result.get('grade',{}))
    except ValueError as exc:
        raise RejectedTrajectory('stale_scoring_version') from exc
    from officegym.harness.isolation import require_isolation
    try:
        require_isolation(root,result)
    except ValueError as exc:
        raise RejectedTrajectory('unsafe_policy_isolation') from exc
    if not result['grade']['full_success'] or result['termination']['state']!='DECLARED_DONE' or not result['operation_audit']['consistent']:
        raise RejectedTrajectory('not_verified_success')
    from .teacher_world import verify_teacher_world
    verify_teacher_world(root,result)
    trace_path = root/'trajectory.jsonl'
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if not records or records[0].get('kind')!='run_header' or 'initial_messages' not in records[0] or 'tool_schemas' not in records[0]:
        raise RejectedTrajectory('missing_original_prompt')
    header = records[0]
    if header.get('harness_version')!=HARNESS_VERSION:
        raise RejectedTrajectory('stale_harness_version')
    if header['mode']!='full' or header['model_version']!=result['model_version']:
        raise RejectedTrajectory('teacher_mode_or_model_mismatch')
    if any(row['kind']=='context_selection' and (row.get('compression_triggered') or row.get('compressed') or row.get('summary_output_tokens',0) or row.get('removed_groups',0)) for row in records):
        raise RejectedTrajectory('compressed_or_truncated')
    if any(row.get('reconciled') or row.get('retry_same_key') for row in records):
        raise RejectedTrajectory('recovered')
    if any(row['kind']=='parse_failure' for row in records):
        raise RejectedTrajectory('parse_retried_source_not_losslessly_reconstructable')
    messages = list(header['initial_messages'])
    pending = {}
    response = None
    for row in records[1:]:
        if row['kind']=='model_response':
            if response is not None or row.get('raw') is None:
                raise RejectedTrajectory('missing_or_unmatched_original_response')
            response = row
        elif row['kind']=='action_start':
            if response is None or any(row.get(field)!=response.get(field) for field in ('tool','arguments')):
                raise RejectedTrajectory('action_differs_from_original_response')
            if row['action_id'] in pending:
                raise RejectedTrajectory('duplicate_action_identity')
            pending[row['action_id']] = (row,response)
            response = None
        elif row['kind']=='interaction':
            if row.get('action_id') not in pending:
                raise RejectedTrajectory('unmatched_interaction')
            action,original = pending.pop(row['action_id'])
            pair = row['messages']
            if len(pair)!=2 or pair[0].get('tool_call',{}).get('id')!=pair[1].get('tool_call_id'):
                raise RejectedTrajectory('broken_tool_pair')
            call = pair[0]['tool_call']
            if call.get('name')!=action['tool'] or call.get('arguments')!=action['arguments'] or pair[0].get('content')!=original.get('content'):
                raise RejectedTrajectory('interaction_differs_from_original_response')
            messages.extend(pair)
        elif row['kind']=='parse_failure' and row.get('message'):
            messages.append(row['message'])
    if pending or response is not None:
        raise RejectedTrajectory('unresolved_operations')
    from .teacher_world import verify_teacher_trace
    verify_teacher_trace(root,records)
    trajectory = dict(instance_id=result['instance_id'],split='train',grade=result['grade'],messages=messages,tools=header['tool_schemas'],
                      model_version=result['model_version'],harness_version=HARNESS_VERSION,compacted=False,recovered=False,source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
    trajectory['conversation'] = render_messages(messages)
    trajectory['content_sha256'] = hashlib.sha256(canonical(trajectory).encode()).hexdigest()
    return trajectory

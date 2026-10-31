"""Evidence-backed audit ledger; authored fixtures never count as model discoveries."""
import hashlib
import json
import os
from pathlib import Path
from officegym.env.world import canonical


def append_finding(ledger_path,*,finding_id,root_cause,source_kind,source_run,minimal_reproduction,old_grade,new_grade,affected_checkpoints,notes):
    if source_kind not in ('model_trajectory','authored_counterexample'):
        raise ValueError('explicit finding source required')
    if not all(isinstance(value,str) and value.strip() for value in (finding_id,root_cause,notes)):
        raise ValueError('finding identity, root cause and notes required')
    ledger = Path(ledger_path)
    prior = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    if any(row['finding_id']==finding_id for row in prior):
        raise ValueError('duplicate finding identity')
    reproduction = Path(minimal_reproduction).resolve()
    if not reproduction.is_file():
        raise ValueError('minimal reproduction artifact missing')
    source = Path(source_run).resolve()
    trace = source/'trajectory.jsonl' if source_kind=='model_trajectory' else source/'actions.jsonl'
    if not trace.is_file():
        raise ValueError('source trajectory missing')
    if source_kind=='model_trajectory':
        result = json.loads((source/'result.json').read_text())
        if result.get('policy_kind')!='inference_service' or not result.get('model_version'):
            raise ValueError('model discovery requires genuine model run provenance')
        records = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        if not any(row['kind']=='model_response' and row.get('raw') is not None for row in records):
            raise ValueError('original model response missing')
    if not isinstance(affected_checkpoints,list) or any(not isinstance(item,str) or not item for item in affected_checkpoints):
        raise ValueError('affected checkpoint hashes required')
    if old_grade.get('eval_grader_version')==new_grade.get('eval_grader_version'):
        raise ValueError('grader revision must change in a fix audit')
    row = dict(finding_id=finding_id,root_cause=root_cause,source_kind=source_kind,source_run=str(source),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
               minimal_reproduction=str(reproduction),reproduction_sha256=hashlib.sha256(reproduction.read_bytes()).hexdigest(),old_grade=old_grade,new_grade=new_grade,
               affected_checkpoints=affected_checkpoints,notes=notes,reevaluation_status='pending' if affected_checkpoints else 'not_applicable_no_checkpoints')
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a') as stream:
        stream.write(canonical(row)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row


def summarize(ledger_path):
    path = Path(ledger_path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    model = {row['root_cause'] for row in rows if row['source_kind']=='model_trajectory'}
    authored = {row['root_cause'] for row in rows if row['source_kind']=='authored_counterexample'}
    return {'model_observed_root_causes':len(model),'authored_counterexample_root_causes':len(authored), 'findings':len(rows),
            'pending_reevaluations':sum(row['reevaluation_status']=='pending' for row in rows)}

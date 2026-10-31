"""Bounded serial evaluation of an existing hashed plan, without silent drops."""
import hashlib
import json
import os
from pathlib import Path
from officegym.env.world import canonical
from officegym.tasks.validate import validate_dataset
from .configured import evaluate_instance,validate_config
from .export import result_row
from .plan import run_key,missing_runs,P1


def evaluate_batch(manifest_path,plan_path,config_path,out,*,group,split,max_runs,max_generated_tokens,execute=evaluate_instance,resume=False):
    if group not in P1 or split not in ('dev','id_test','combo_test','unseen_test'):
        raise ValueError('invalid planned experiment selection')
    if type(max_runs) is not int or max_runs<1 or type(max_generated_tokens) is not int or max_generated_tokens<1:
        raise ValueError('explicit finite batch budgets required')
    manifest_path,plan_path,config_path = map(lambda p:Path(p).resolve(),(manifest_path,plan_path,config_path))
    validate_dataset(manifest_path)
    fingerprint = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if Path(str(plan_path)+'.sha256').read_text().strip()!=fingerprint:
        raise ValueError('frozen plan checksum mismatch')
    plan = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    missing_runs(plan,[])  # reject duplicate keys before any inference
    selected = [row for row in plan if row['group']==group and row['split']==split]
    if not selected:
        raise ValueError('no matching planned runs')
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config['mode']!=('simple' if group.endswith('_simple') else 'full'):
        raise ValueError('config mode differs from experiment')
    entries = {entry['instance_id']:entry for entry in json.loads(manifest_path.read_text())['instances']}
    for row in selected:
        if row['instance_id'] not in entries:
            raise ValueError('planned instance missing')
        if any(entries[row['instance_id']][key]!=row[key] for key in ('split','scenario_family_id','template_id','depth_level','perturbation_level','dependency_width')):
            raise ValueError('plan dataset metadata mismatch')
    root = Path(out)
    selection = dict(group=group,split=split,plan_sha256=fingerprint,max_runs=max_runs,max_generated_tokens=max_generated_tokens,
                     config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    if resume:
        if json.loads((root/'selection.json').read_text())!=selection:
            raise ValueError('resume configuration, dataset, plan or budgets changed')
    else:
        root.mkdir(parents=True,exist_ok=False)
        (root/'selection.json').write_text(canonical(selection)+'\n')
    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    results = read_rows(root/'results.jsonl')
    missing_runs(selected,results)
    completed = {run_key(row) for row in results}
    reservations = read_rows(root/'reservations.jsonl')
    reserved_keys = {tuple(row['run_key']) for row in reservations}
    if len(reserved_keys)!=len(reservations):
        raise ValueError('duplicate run reservation')
    if any(key not in {run_key(row) for row in selected} for key in reserved_keys):
        raise ValueError('unplanned budget reservation')
    reserved = sum(row['reserved_tokens'] for row in reservations)
    cap = config['budget']['generated_tokens']
    with (root/'results.jsonl').open('a') as ledger, (root/'reservations.jsonl').open('a') as reservation_log:
        for planned in selected:
            key = run_key(planned)
            if key in completed:
                continue
            identity = hashlib.sha256(canonical(key).encode()).hexdigest()[:24]
            directory = root/identity
            if key in reserved_keys:
                if not (directory/'result.json').is_file():
                    raise ValueError('interrupted reserved run has no terminal result; retain artifacts for explicit recovery')
            else:
                if len(reserved_keys)>=max_runs or reserved+cap>max_generated_tokens:
                    break
                reservation_log.write(canonical({'run_key':key,'reserved_tokens':cap})+'\n')
                reservation_log.flush()
                os.fsync(reservation_log.fileno())
                reserved += cap
                reserved_keys.add(key)
                execute(manifest_path.parent/entries[planned['instance_id']]['path'],directory,config_path)
            row = result_row(directory,planned,manifest_path)
            row['run_plan_sha256'] = fingerprint
            ledger.write(canonical(row)+'\n')
            ledger.flush()
            os.fsync(ledger.fileno())
            results.append(row)
            completed.add(key)
    absent = missing_runs(selected,results)
    (root/'missing_runs.jsonl').write_text(''.join(canonical(row)+'\n' for row in absent))
    summary = dict(planned=len(selected),completed=len(results),missing=len(absent),reserved_generated_tokens=reserved,complete=not absent)
    (root/'summary.json').write_text(canonical(summary)+'\n')
    return summary

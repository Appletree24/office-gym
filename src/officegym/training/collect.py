"""Finite teacher collection with retained failures and conservative reservations."""
import json
import os
import time
import math
import hashlib
from pathlib import Path
from officegym.env.world import canonical
from officegym.eval.artifacts import write_json
from officegym.tasks.validate import validate_dataset
from officegym.eval.configured import evaluate_instance,validate_config
from .pilot import select_training_tasks,selection_record,DEFAULT_SEED


def collect(manifest_path,config_path,out,*,max_attempts_per_task,max_episodes,max_generated_tokens,run=evaluate_instance,max_tasks=None,selection_seed=DEFAULT_SEED):
    if any(type(value) is not int or value<1 for value in (max_attempts_per_task,max_episodes,max_generated_tokens)) or max_attempts_per_task>3:
        raise ValueError('explicit finite collection budgets and at most three attempts required')
    manifest_path,config_path = Path(manifest_path).resolve(),Path(config_path).resolve()
    validate_dataset(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest['instances'] if entry['split']=='train']
    if not entries:
        raise ValueError('no training instances')
    if max_tasks is not None:
        entries=select_training_tasks(manifest,max_tasks,selection_seed)
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config['mode']!='full':
        raise ValueError('teacher collection requires full harness')
    episode_cap = config['budget']['generated_tokens']
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    selection=selection_record(manifest,entries,hashlib.sha256(manifest_path.read_bytes()).hexdigest(),max_tasks,selection_seed) if max_tasks is not None else dict(schema='officegym-teacher-selection-v1',selection_method='all-training-tasks-manifest-order',selected_tasks=len(entries),source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),tasks=[dict(instance_id=row['instance_id'],path=row['path'],split=row['split']) for row in entries])
    write_json(root/'selection.json',selection)
    frozen_config=root/'resolved_sampling_config.json'
    write_json(frozen_config,config)
    write_json(root/'source.json',{'dataset_manifest':str(manifest_path),'dataset_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(frozen_config.read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()})
    attempts,successes = [],[]
    reserved = observed = 0
    stopped = False
    with (root/'attempts.jsonl').open('w') as ledger, (root/'reservations.jsonl').open('w') as reservations:
        for entry in entries:
            for attempt in range(max_attempts_per_task):
                if len(attempts)>=max_episodes or reserved+episode_cap>max_generated_tokens:
                    stopped = True
                    break
                directory = root/(entry['instance_id']+'_attempt_'+str(attempt))
                # Reserve the complete cap even if usage is missing after a failure.
                reservations.write(canonical({'instance_id':entry['instance_id'],'attempt_id':attempt,'run_dir':str(directory),'reserved_generated_tokens':episode_cap})+'\n')
                reservations.flush()
                os.fsync(reservations.fileno())
                reserved += episode_cap
                started=time.monotonic()
                execution_error = None
                try:
                    result = run(manifest_path.parent/entry['path'],directory,frozen_config)
                except Exception as exc:
                    execution_error = type(exc).__name__
                    directory.mkdir(parents=True,exist_ok=True)
                    (directory/'collection_error.json').write_text(canonical({'error_type':execution_error,'usage_unknown':True})+'\n')
                    result = {'termination':{'state':'INFRA_ERROR'},'grade':{'full_success':False}}
                elapsed=time.monotonic()-started
                def count(name):
                    value=result['termination'].get(name)
                    return value if type(value) is int and value>=0 else None
                tokens=count('generated_tokens')
                if tokens is not None:
                    observed += tokens
                overrun=tokens is not None and tokens>episode_cap
                if overrun:
                    stopped=True
                reported_wall=result.get('runner_wall_seconds',result['termination'].get('wall_seconds'))
                if type(reported_wall) not in (int,float) or not math.isfinite(reported_wall) or reported_wall<0:
                    reported_wall=None
                row = dict(instance_id=entry['instance_id'],attempt_id=attempt,run_dir=str(directory),success=bool(result['grade']['full_success']) and not overrun,reported_input_tokens=count('input_tokens'),reported_tool_calls=count('tool_calls'),reported_wall_seconds=reported_wall,collector_wall_seconds=elapsed,generation_cap_overrun=overrun,state=result['termination']['state'],reported_generated_tokens=tokens,reserved_generated_tokens=episode_cap,execution_error_type=execution_error)
                attempts.append(row)
                ledger.write(canonical(row)+'\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                if overrun:
                    break
                if row['success']:
                    successes.append(str(directory))
                    break
            if stopped:
                break
    (root/'successful_runs.json').write_text(canonical(successes)+'\n')
    resource_totals={field:{'reported_total':sum(row[field] for row in attempts if row[field] is not None),'unknown_attempts':sum(row[field] is None for row in attempts)} for field in ('reported_generated_tokens','reported_input_tokens','reported_tool_calls','reported_wall_seconds')}
    summary = dict(resources=resource_totals,usage_complete=all(value['unknown_attempts']==0 for value in resource_totals.values()),collector_wall_seconds=sum(row['collector_wall_seconds'] for row in attempts),generation_cap_overruns=sum(row['generation_cap_overrun'] for row in attempts),gpu_hours=None,api_cost_usd=None,attempts=len(attempts),successful_tasks=len(successes),reported_generated_tokens=observed,reserved_generated_tokens=reserved,
                   selected_tasks=len(entries),attempted_tasks=len({row['instance_id'] for row in attempts}),max_tasks=max_tasks,budget_stop=stopped,max_episodes=max_episodes,max_generated_tokens=max_generated_tokens,max_attempts_per_task=max_attempts_per_task,
                   note='Reservations include unused capacity and unknown failure usage, not measured cost. Reported totals are lower bounds when unknown_attempts is positive; null GPU/API cost is unmeasured.')
    write_json(root/'summary.json',summary)
    return summary

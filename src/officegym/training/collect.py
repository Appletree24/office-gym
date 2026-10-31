"""Finite teacher collection with retained failures and conservative reservations."""
import json
import os
from pathlib import Path
from officegym.env.world import canonical
from officegym.tasks.validate import validate_dataset
from officegym.eval.configured import evaluate_instance,validate_config


def collect(manifest_path,config_path,out,*,max_attempts_per_task,max_episodes,max_generated_tokens,run=evaluate_instance):
    if any(type(value) is not int or value<1 for value in (max_attempts_per_task,max_episodes,max_generated_tokens)) or max_attempts_per_task>3:
        raise ValueError('explicit finite collection budgets and at most three attempts required')
    manifest_path,config_path = Path(manifest_path).resolve(),Path(config_path).resolve()
    validate_dataset(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest['instances'] if entry['split']=='train']
    if not entries:
        raise ValueError('no training instances')
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config['mode']!='full':
        raise ValueError('teacher collection requires full harness')
    episode_cap = config['budget']['generated_tokens']
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    attempts,successes = [],[]
    reserved = observed = 0
    stopped = False
    with (root/'attempts.jsonl').open('w') as ledger:
        for entry in entries:
            for attempt in range(max_attempts_per_task):
                if len(attempts)>=max_episodes or reserved+episode_cap>max_generated_tokens:
                    stopped = True
                    break
                directory = root/(entry['instance_id']+'_attempt_'+str(attempt))
                # Reserve the complete cap even if usage is missing after a failure.
                reserved += episode_cap
                execution_error = None
                try:
                    result = run(manifest_path.parent/entry['path'],directory,config_path)
                except Exception as exc:
                    execution_error = type(exc).__name__
                    directory.mkdir(parents=True,exist_ok=True)
                    (directory/'collection_error.json').write_text(canonical({'error_type':execution_error,'usage_unknown':True})+'\n')
                    result = {'termination':{'state':'INFRA_ERROR'},'grade':{'full_success':False}}
                tokens = result['termination'].get('generated_tokens')
                if type(tokens) is int:
                    observed += tokens
                row = dict(instance_id=entry['instance_id'],attempt_id=attempt,run_dir=str(directory),success=bool(result['grade']['full_success']),state=result['termination']['state'],reported_generated_tokens=tokens,reserved_generated_tokens=episode_cap,execution_error_type=execution_error)
                attempts.append(row)
                ledger.write(canonical(row)+'\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                if row['success']:
                    successes.append(str(directory))
                    break
            if stopped:
                break
    (root/'successful_runs.json').write_text(canonical(successes)+'\n')
    summary = dict(attempts=len(attempts),successful_tasks=len(successes),reported_generated_tokens=observed,reserved_generated_tokens=reserved,
                   budget_stop=stopped,max_episodes=max_episodes,max_generated_tokens=max_generated_tokens,max_attempts_per_task=max_attempts_per_task,
                   note='Reservation budget includes unused capacity and unknown failure usage; not measured cost.')
    (root/'summary.json').write_text(canonical(summary)+'\n')
    return summary

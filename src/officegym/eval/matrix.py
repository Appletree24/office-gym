"""Serial frozen-plan matrix with persisted conservative global reservations."""
import fcntl
import json
import os
from pathlib import Path
from officegym.env.world import canonical
from officegym.tasks.freeze import digest,verify_frozen_dataset
from .batch import evaluate_batch
from .configured import validate_config
from .export import evaluation_budget_hash
from .plan import P0,P1,build_plan,missing_runs


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def evaluate_matrix(manifest_path,plan_path,configs_path,freeze_path,out,*,priority='P0',splits=('id_test','combo_test','unseen_test'),max_runs,max_generated_tokens,resume=False,execute_batch=evaluate_batch):
    if priority not in ('P0','P1') or not splits or len(set(splits))!=len(splits) or not set(splits)<={'dev','id_test','combo_test','unseen_test'}:
        raise ValueError('valid unique matrix priority/splits required')
    if type(max_runs) is not int or max_runs<1 or type(max_generated_tokens) is not int or max_generated_tokens<1:
        raise ValueError('explicit finite matrix budgets required')
    manifest,plan_path,configs_path,freeze_path=map(lambda p:Path(p).resolve(),(manifest_path,plan_path,configs_path,freeze_path))
    verify_frozen_dataset(freeze_path,manifest)
    plan_hash=digest(plan_path)
    if Path(str(plan_path)+'.sha256').read_text().strip()!=plan_hash:raise ValueError('matrix run plan checksum mismatch')
    plan=read_rows(plan_path);groups=P0 if priority=='P0' else P1
    expected=build_plan(json.loads(manifest.read_text())['instances'],priority=priority,splits=splits)
    if plan!=expected:raise ValueError('matrix requires the complete predeclared priority/split plan')
    mapping=json.loads(configs_path.read_text())
    if not isinstance(mapping,dict) or set(mapping)!=set(groups):raise ValueError('one explicit config for every matrix group required')
    configs={};paths={}
    for group in groups:
        if not isinstance(mapping[group],str) or not mapping[group].strip():raise ValueError('matrix config path required')
        path=(configs_path.parent/mapping[group]).resolve();config=json.loads(path.read_text());validate_config(config)
        if config['mode']!=('simple' if group.endswith('_simple') else 'full'):raise ValueError('matrix config mode differs from group')
        configs[group]=config;paths[group]=path
    if len({evaluation_budget_hash(config['budget'],config) for config in configs.values()})!=1:
        raise ValueError('matrix comparisons require one shared evaluation budget')
    for simple,full in (('api_simple','api_full'),('base_simple','base_full'),('grpo_simple','grpo_full')):
        if simple in configs and any(configs[simple][key]!=configs[full][key] for key in ('model','model_version','tokenizer_path','tokenizer_revision')):
            raise ValueError('paired matrix groups require the same model/tokenizer')
    inputs={str(path):digest(path) for path in {manifest,plan_path,Path(str(plan_path)+'.sha256'),configs_path,freeze_path,Path(str(freeze_path)+'.sha256'),*paths.values()}}
    identity=dict(priority=priority,splits=list(splits),max_runs=max_runs,max_generated_tokens=max_generated_tokens,inputs_sha256=inputs)
    root=Path(out)
    if not resume:root.mkdir(parents=True,exist_ok=False)
    with (root/'matrix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            if json.loads((root/'matrix_selection.json').read_text())!=identity:raise ValueError('matrix resume inputs or budgets changed')
        else:
            with (root/'matrix_selection.json').open('x') as stream:stream.write(canonical(identity)+'\n')
        allocations=read_rows(root/'matrix_reservations.jsonl')
        keys={(r['group'],r['split']) for r in allocations}
        if len(keys)!=len(allocations):raise ValueError('duplicate matrix allocation')
        selected={(group,split) for group in groups for split in splits}
        for row in allocations:
            if (row['group'],row['split']) not in selected or type(row['max_runs']) is not int or row['max_runs']<1 or row['max_generated_tokens']!=row['max_runs']*configs[row['group']]['budget']['generated_tokens']:
                raise ValueError('invalid persisted matrix allocation')
        if sum(r['max_runs'] for r in allocations)>max_runs or sum(r['max_generated_tokens'] for r in allocations)>max_generated_tokens:
            raise ValueError('persisted matrix reservations exceed global budgets')
        def snapshot():
            rows=[]
            for group in groups:
                for split in splits:
                    batch_rows=read_rows(root/group/split/'results.jsonl')
                    allocation=next((r for r in allocations if (r['group'],r['split'])==(group,split)),None)
                    if batch_rows and (allocation is None or len(batch_rows)>allocation['max_runs']):raise ValueError('matrix rows exceed reserved run slots')
                    if any(r['group']!=group or r['split']!=split for r in batch_rows):raise ValueError('matrix batch row identity mismatch')
                    rows.extend(batch_rows)
            absent=missing_runs(plan,rows)
            (root/'results.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
            (root/'missing_runs.jsonl').write_text(''.join(canonical(r)+'\n' for r in absent))
            summary=dict(planned=len(plan),completed=len(rows),missing=len(absent),complete=not absent,reserved_run_slots=sum(r['max_runs'] for r in allocations),reserved_generated_tokens=sum(r['max_generated_tokens'] for r in allocations),scope='Conservative matrix allocations; unused reserved slots are not recycled. Missing runs are retained, not averaged away.')
            (root/'summary.json').write_text(canonical(summary)+'\n')
            return summary
        snapshot()
        with (root/'matrix_reservations.jsonl').open('a') as ledger:
            for group in groups:
                for split in splits:
                    if any(digest(path)!=value for path,value in inputs.items()):raise ValueError('matrix inputs changed during execution')
                    allocation=next((r for r in allocations if (r['group'],r['split'])==(group,split)),None)
                    if allocation is None:
                        cap=configs[group]['budget']['generated_tokens']
                        remaining_runs=max_runs-sum(r['max_runs'] for r in allocations)
                        remaining_tokens=max_generated_tokens-sum(r['max_generated_tokens'] for r in allocations)
                        count=min(sum(r['group']==group and r['split']==split for r in plan),remaining_runs,remaining_tokens//cap)
                        if count<1:continue
                        allocation=dict(group=group,split=split,max_runs=count,max_generated_tokens=count*cap)
                        ledger.write(canonical(allocation)+'\n');ledger.flush();os.fsync(ledger.fileno());allocations.append(allocation)
                    directory=root/group/split
                    try:
                        execute_batch(manifest,plan_path,paths[group],directory,group=group,split=split,max_runs=allocation['max_runs'],max_generated_tokens=allocation['max_generated_tokens'],resume=directory.exists(),dataset_freeze=freeze_path)
                    except BaseException as error:
                        snapshot()
                        (root/'last_failure.json').write_text(canonical(dict(group=group,split=split,error_type=type(error).__name__,scope='No successful row inferred for interrupted/failed execution.'))+'\n')
                        raise
                    snapshot()
        return snapshot()

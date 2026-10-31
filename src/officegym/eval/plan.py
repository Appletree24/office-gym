"""Freeze planned run keys before evaluation; never silently discard missing runs."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical

P0 = ('api_simple','api_full','base_full','sft_full','grpo_full')
P1 = P0 + ('base_simple','grpo_simple','rs_sft_full')
FIELDS = ('split','instance_id','scenario_family_id','template_id','depth_level','perturbation_level','dependency_width')


def run_key(row):
    return tuple(row[key] for key in ('group','split','instance_id','repeat_id'))


def build_plan(instances,priority='P0',splits=('id_test','combo_test','unseen_test')):
    if priority not in ('P0','P1'):
        raise ValueError('unknown experiment priority')
    if not set(splits) <= {'dev','id_test','combo_test','unseen_test'}:
        raise ValueError('invalid evaluation split')
    groups = P0 if priority=='P0' else P1
    rows, identities = [], set()
    for instance in instances:
        if instance['instance_id'] in identities:
            raise ValueError('duplicate instance identity')
        identities.add(instance['instance_id'])
        if instance['split'] not in splits:
            continue
        if instance.get('example_only'):
            raise ValueError('example instance cannot enter run plan')
        base = {key:instance[key] for key in FIELDS}
        if any(not isinstance(base[key],str) or not base[key] for key in FIELDS[:4]):
            raise ValueError('missing task identity metadata')
        if type(base['depth_level']) is not int or base['depth_level'] not in (1,2,3) or type(base['perturbation_level']) is not int or base['perturbation_level'] not in (0,1,2):
            raise ValueError('invalid experimental condition')
        if type(base['dependency_width']) is not int or base['dependency_width']<1:
            raise ValueError('positive measured dependency width required')
        for group in groups:
            for repeat in (0,1,2):
                rows.append(dict(base,group=group,repeat_id=repeat))
    if not rows or set(splits)-{row['split'] for row in rows}:
        raise ValueError('requested split has no instances')
    return sorted(rows,key=run_key)


def freeze_plan(rows,path):
    path = Path(path)
    payload = ''.join(canonical(row)+'\n' for row in rows).encode()
    fingerprint = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
    with Path(str(path)+'.sha256').open('x') as stream:
        stream.write(fingerprint+'\n')
    return fingerprint


def missing_runs(plan,results):
    planned = {run_key(row):row for row in plan}
    if len(planned)!=len(plan):
        raise ValueError('duplicate planned run')
    completed = set()
    for result in results:
        key = run_key(result)
        if key not in planned:
            raise ValueError('unplanned result')
        if key in completed:
            raise ValueError('duplicate final result; attempts require separate ledger')
        if any(result.get(field)!=planned[key][field] for field in FIELDS):
            raise ValueError('result metadata differs from plan')
        completed.add(key)
    return [planned[key] for key in sorted(set(planned)-completed)]

"""Outcome-independent training-only teacher pilot selection; never calls a model."""
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.tasks.validate import validate_dataset

DEFAULT_SEED='officegym-teacher-pilot-v1'


def select_training_tasks(manifest,max_tasks,seed=DEFAULT_SEED):
    if type(max_tasks) is not int or max_tasks<1:raise ValueError('positive finite task cap required')
    if not isinstance(seed,str) or not seed:raise ValueError('explicit selection seed required')
    entries=[entry for entry in manifest['instances'] if entry['split']=='train']
    if not entries:raise ValueError('no training instances')
    ids=[entry['instance_id'] for entry in entries]
    if len(set(ids))!=len(ids):raise ValueError('duplicate training identity')
    buckets=defaultdict(lambda:defaultdict(list))
    for entry in entries:
        buckets[entry['template_id']][(entry['depth_level'],entry['perturbation_level'])].append(entry)
    queues={}
    for template,strata in sorted(buckets.items()):
        pools=[]
        for stratum,rows in sorted(strata.items()):
            # Selection sees identifiers/strata only, never grades or reference lengths.
            rows=sorted(rows,key=lambda row:hashlib.sha256(canonical([seed,row['instance_id']]).encode()).hexdigest())
            pools.append(deque(rows))
        ordered=[]
        while any(pools):
            for pool in pools:
                if pool:ordered.append(pool.popleft())
        queues[template]=deque(ordered)
    selected=[]
    while len(selected)<min(max_tasks,len(entries)):
        for queue in queues.values():
            if queue:selected.append(queue.popleft())
            if len(selected)==min(max_tasks,len(entries)):break
    return selected


def selection_record(manifest,entries,manifest_sha256,max_tasks,seed):
    return dict(schema='officegym-teacher-selection-v1',source_manifest_sha256=manifest_sha256,
                selection_method='template-round-robin/stratum-round-robin/sha256-seeded-identity',
                seed=seed,max_tasks=max_tasks,available_train_tasks=sum(row['split']=='train' for row in manifest['instances']),
                selected_tasks=len(entries),template_counts=dict(sorted(Counter(row['template_id'] for row in entries).items())),
                tasks=[{key:row[key] for key in ('instance_id','path','split','template_id','depth_level','perturbation_level')} for row in entries],
                scope='Training-only pilot plan; no teacher outcomes, model calls, price estimate or frozen data admission.')


def plan(manifest_path,out,*,max_tasks=40,seed=DEFAULT_SEED):
    source=Path(manifest_path).resolve();validate_dataset(source)
    manifest=json.loads(source.read_text());entries=select_training_tasks(manifest,max_tasks,seed)
    record=selection_record(manifest,entries,hashlib.sha256(source.read_bytes()).hexdigest(),max_tasks,seed)
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    (root/'selection.json').write_text(canonical(record)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--max-tasks',type=int,default=40);parser.add_argument('--seed',default=DEFAULT_SEED)
    args=parser.parse_args();print(json.dumps(plan(args.manifest,args.out,max_tasks=args.max_tasks,seed=args.seed),indent=2))


if __name__=='__main__':main()

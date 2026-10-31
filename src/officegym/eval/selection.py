"""Freeze checkpoint candidates before selecting from complete dev-only results."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.training.checkpoints import checkpoint_fingerprint
from .analysis import validate_row,VERSION_FIELDS,TASK_FIELDS
from .plan import missing_runs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def frozen(path):
    if Path(str(path)+'.sha256').read_text().strip()!=digest(path):
        raise ValueError('frozen selection artifact checksum mismatch')


def freeze(candidates,plan_path,group,out):
    stages={'sft_full':'sft','grpo_full':'grpo','rs_sft_full':'rs_sft'}
    if group not in stages:
        raise ValueError('selection requires a full-harness training group')
    plan_path=Path(plan_path).resolve()
    frozen(plan_path)
    plan=rows(plan_path)
    if any(row.get('synthetic') or row.get('example_only') for row in plan):
        raise ValueError('synthetic selection plans are not accepted')
    if any(row['split']!='dev' for row in plan):
        raise ValueError('checkpoint selection plan must be dev-only')
    selected=[row for row in plan if row['group']==group]
    missing_runs(selected,[])
    if not selected:
        raise ValueError('no selected dev runs')
    for instance in {row['instance_id'] for row in selected}:
        instance_rows=[row for row in selected if row['instance_id']==instance]
        if any(len({row[field] for row in instance_rows})!=1 for field in TASK_FIELDS):
            raise ValueError('inconsistent task metadata in dev plan')
        if {row['repeat_id'] for row in instance_rows}!={0,1,2}:
            raise ValueError('dev selection requires all three planned repeats')
    bound=[]
    for candidate in candidates:
        step=candidate['optimizer_step']
        if type(step) is not int or step<0:
            raise ValueError('nonnegative candidate optimizer step required')
        export=Path(candidate['export_record']).resolve()
        record=json.loads(export.read_text())
        checkpoint=Path(candidate['checkpoint']).resolve()
        fingerprint=checkpoint_fingerprint(checkpoint)['checkpoint_sha256']
        if record.get('stage')!=stages[group] or record.get('checkpoint_sha256')!=fingerprint or not record.get('equivalence_report_sha256'):
            raise ValueError('candidate must be an equivalence-verified export of the selected stage')
        bound.append(dict(candidate_id=candidate['candidate_id'],optimizer_step=step,checkpoint=str(checkpoint),checkpoint_sha256=fingerprint,export_record=str(export),export_sha256=digest(export)))
    if not bound or any(not isinstance(row['candidate_id'],str) or not row['candidate_id'] for row in bound) or len({row['candidate_id'] for row in bound})!=len(bound) or len({row['checkpoint_sha256'] for row in bound})!=len(bound):
        raise ValueError('nonempty unique checkpoint candidates required')
    rule=dict(format='officegym_dev_selection_v1',group=group,dev_plan=str(plan_path),dev_plan_sha256=digest(plan_path),candidates=bound,
              ranking=['highest_instance_macro_success','lowest_optimizer_step','checkpoint_sha256'])
    path=Path(out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(canonical(rule)+'\n')
    with Path(str(path)+'.sha256').open('x') as stream:
        stream.write(digest(path)+'\n')
    return rule


def select(rule_path,result_files,out):
    frozen(rule_path)
    rule=json.loads(Path(rule_path).read_text())
    if rule.get('format')!='officegym_dev_selection_v1' or rule.get('ranking')!=['highest_instance_macro_success','lowest_optimizer_step','checkpoint_sha256']:
        raise ValueError('unsupported frozen checkpoint selection rule')
    frozen(rule['dev_plan'])
    if digest(rule['dev_plan'])!=rule['dev_plan_sha256']:
        raise ValueError('dev plan changed after selection rule freeze')
    if set(result_files)!={candidate['candidate_id'] for candidate in rule['candidates']}:
        raise ValueError('complete result files for all frozen candidates required')
    plan=[row for row in rows(rule['dev_plan']) if row['group']==rule['group']]
    versions=set()
    scores=[]
    for candidate in rule['candidates']:
        if digest(candidate['export_record'])!=candidate['export_sha256'] or checkpoint_fingerprint(candidate['checkpoint'])['checkpoint_sha256']!=candidate['checkpoint_sha256']:
            raise ValueError('checkpoint candidate changed after freeze')
        path=Path(result_files[candidate['candidate_id']]).resolve()
        measured=rows(path)
        for row in measured:
            validate_row(row)
            if row['split']!='dev' or row['group']!=rule['group'] or row['model_version']!=candidate['checkpoint_sha256']:
                raise ValueError('selection results must match dev group and checkpoint identity')
            versions.add(tuple(row[field] for field in VERSION_FIELDS))
        if missing_runs(plan,measured):
            raise ValueError('missing dev runs for checkpoint candidate')
        per_instance={identity:[row['success'] for row in measured if row['instance_id']==identity] for identity in {row['instance_id'] for row in measured}}
        score=sum(sum(values)/len(values) for values in per_instance.values())/len(per_instance)
        scores.append(dict(candidate,dev_macro_success=score,dev_instances=len(per_instance),dev_runs=len(measured),results_path=str(path),results_sha256=digest(path)))
    if len(versions)!=1:
        raise ValueError('incomparable dev scoring, dataset, harness or budgets')
    ranked=sorted(scores,key=lambda row:(-row['dev_macro_success'],row['optimizer_step'],row['checkpoint_sha256']))
    result=dict(selection_rule_sha256=digest(rule_path),selected=ranked[0],all_candidates=scores,scope='dev_checkpoint_selection_not_test_performance')
    with Path(out).open('x') as stream:
        stream.write(canonical(result)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    initial=sub.add_parser('freeze')
    for field in ('candidates','plan','group','out'):
        initial.add_argument('--'+field,required=True)
    finish=sub.add_parser('select')
    for field in ('rule','results','out'):
        finish.add_argument('--'+field,required=True)
    args=parser.parse_args()
    if args.command=='freeze':
        result=freeze(json.loads(Path(args.candidates).read_text()),args.plan,args.group,args.out)
    else:
        result=select(args.rule,json.loads(Path(args.results).read_text()),args.out)
    print(json.dumps(result,indent=2))

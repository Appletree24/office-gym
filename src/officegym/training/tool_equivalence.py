"""Prospective paired dev tool-behavior gate for a numerical export check."""
import json
import math
from fractions import Fraction
from pathlib import Path
from officegym.env.world import canonical
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
from officegym.harness.versions import HARNESS_VERSION
from officegym.tasks.validate import validate_dataset
from officegym.eval.analysis import validate_row,VERSION_FIELDS
from officegym.eval.export import result_row
from officegym.eval.plan import missing_runs,run_key,FIELDS
from .equivalence import validate as validate_numeric,digest,replay_comparisons,replay_control


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def numeric_evidence(report_path,plan_path):
    report=json.loads(Path(report_path).read_text());plan=json.loads(Path(plan_path).read_text())
    _,merged,probes=validate_numeric(plan)
    if report.get('plan_sha256')!=digest(plan_path) or report.get('passed_numeric') is not True or report.get('exported_checkpoint_sha256')!=merged['exported_checkpoint_sha256'] or report.get('source_checkpoint_files_sha256')!=merged['training_source']['source_checkpoint_files_sha256'] or report.get('optimizer_step')!=merged['training_source']['optimizer_step']:
        raise ValueError('numerical evidence is failed or bound to another export/source')
    if report.get('logit_positions')!='last_input_token_full_vocabulary' or report.get('thresholds')!={'logit_atol':plan['logit_atol'],'logit_rtol':plan.get('logit_rtol',0),'min_generation_match_rate':plan['min_generation_match_rate']}:
        raise ValueError('numerical report scope/thresholds differ from frozen plan')
    vocab=json.loads((Path(merged['model_path'])/'config.json').read_text())['vocab_size']
    comparisons=replay_comparisons(report,probes,vocab,plan['logit_atol'],plan.get('logit_rtol',0))
    if not replay_control(report,probes,vocab,plan['logit_atol'],plan.get('logit_rtol',0),plan.get('negative_control')):raise ValueError('negative control was not detected')
    generation=sum(row['generation_equal'] for row in comparisons)/len(comparisons)
    if report.get('generation_match_rate')!=generation or generation<plan['min_generation_match_rate'] or any(not row['logits']['within_tolerance'] for row in comparisons):
        raise ValueError('numerical metrics do not satisfy frozen thresholds')
    records=report['before']+report['after']
    if plan.get('negative_control'):records+=report['negative_control']['records']
    if any(len(row['generated_ids'])>plan['max_new_tokens'] for row in records):raise ValueError('numerical generation budget exceeded')
    return report


def freeze(spec,out):
    for key in ('max_success_rate_delta','max_disagreement_rate','min_success_rate'):
        if type(spec.get(key)) not in (int,float) or not math.isfinite(spec[key]) or not 0<=spec[key]<=1:
            raise ValueError('finite tool-equivalence thresholds required')
    if spec['min_success_rate']==0:
        raise ValueError('tool-equivalence requires positive baseline success')
    if spec['group'] not in ('sft_full','grpo_full','rs_sft_full'):
        raise ValueError('tool equivalence requires a full-harness training group')
    for name in ('numeric_report','numeric_plan','dataset_manifest','dev_plan','before_results','after_results'):
        if not Path(spec[name]).is_absolute():
            raise ValueError('absolute tool-equivalence artifact paths required')
    if any(Path(spec[name]).exists() for name in ('before_results','after_results')):
        raise ValueError('freeze tool-equivalence rules before collecting tool results')
    numeric_evidence(spec['numeric_report'],spec['numeric_plan'])
    validate_dataset(spec['dataset_manifest'])
    if Path(spec['dev_plan']+'.sha256').read_text().strip()!=digest(spec['dev_plan']):
        raise ValueError('dev plan checksum mismatch')
    plan=read_rows(spec['dev_plan'])
    if any(row.get('synthetic') or row.get('example_only') for row in plan):
        raise ValueError('synthetic tool-equivalence plan is not accepted')
    if any(row['split']!='dev' for row in plan):
        raise ValueError('tool-equivalence plan must be dev-only')
    selected=[row for row in plan if row['group']==spec['group']]
    missing_runs(selected,[])
    entries={entry['instance_id']:entry for entry in json.loads(Path(spec['dataset_manifest']).read_text())['instances']}
    if any(row['instance_id'] not in entries or any(row[field]!=entries[row['instance_id']][field] for field in FIELDS) for row in selected):
        raise ValueError('tool-equivalence dev plan differs from dataset')
    if not selected or any({row['repeat_id'] for row in selected if row['instance_id']==identity}!={0,1,2} for identity in {row['instance_id'] for row in selected}):
        raise ValueError('nonempty complete three-repeat dev plan required')
    rule=dict(spec,format='officegym_tool_equivalence_v1',source_hashes={name:digest(spec[name]) for name in ('numeric_report','numeric_plan','dataset_manifest','dev_plan')},
              versions=dict(env_version='env-v1',eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,harness_version=HARNESS_VERSION))
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with Path(out).open('x') as stream:
        stream.write(canonical(rule)+'\n')
    with Path(str(out)+'.sha256').open('x') as stream:
        stream.write(digest(out)+'\n')
    return rule


def compare_tools(before,after,max_delta,max_disagreement,min_success):
    if not before or len(before)!=len(after):
        raise ValueError('complete paired tool results required')
    a=sum(row['success'] for row in before)/len(before)
    b=sum(row['success'] for row in after)/len(after)
    disagreement=sum(left['success']!=right['success'] for left,right in zip(before,after))/len(before)
    clean=not any(row['infra_error'] for row in before+after)
    duplicate_growth=any(right.get('duplicate_effects',0)>left.get('duplicate_effects',0) for left,right in zip(before,after))
    exact_delta=Fraction(abs(sum(row['success'] for row in after)-sum(row['success'] for row in before)),len(before))
    exact_disagreement=Fraction(sum(left['success']!=right['success'] for left,right in zip(before,after)),len(before))
    exact_success=Fraction(min(sum(row['success'] for row in before),sum(row['success'] for row in after)),len(before))
    return dict(new_duplicate_effects=duplicate_growth,before_success_rate=a,after_success_rate=b,success_rate_delta=b-a,disagreement_rate=disagreement,
                infrastructure_clean=clean,passed=clean and not duplicate_growth and exact_success>=Fraction(str(min_success)) and exact_delta<=Fraction(str(max_delta)) and exact_disagreement<=Fraction(str(max_disagreement)))


def finalize(rule_path,out):
    if Path(str(rule_path)+'.sha256').read_text().strip()!=digest(rule_path):
        raise ValueError('tool-equivalence rule checksum mismatch')
    rule=json.loads(Path(rule_path).read_text())
    if rule.get('format')!='officegym_tool_equivalence_v1' or any(digest(rule[name])!=sha for name,sha in rule['source_hashes'].items()):
        raise ValueError('frozen tool-equivalence inputs changed')
    numeric=numeric_evidence(rule['numeric_report'],rule['numeric_plan'])
    validate_dataset(rule['dataset_manifest'])
    manifest=json.loads(Path(rule['dataset_manifest']).read_text())
    entries={entry['instance_id']:entry for entry in manifest['instances']}
    plan=[row for row in read_rows(rule['dev_plan']) if row['group']==rule['group']]
    planned={run_key(row):row for row in plan}
    sides=[];raw_sides=[];versions=set();profiles=set();artifacts=[]
    for name,model in [('before_results',numeric['source_checkpoint_files_sha256']),('after_results',numeric['exported_checkpoint_sha256'])]:
        records=read_rows(rule[name])
        if missing_runs(plan,records):
            raise ValueError('missing planned tool-equivalence runs')
        ordered=sorted(records,key=run_key);raw_results=[]
        for row in ordered:
            validate_row(row)
            if row['split']!='dev' or row['model_version']!=model or any(row[key]!=value for key,value in rule['versions'].items()):
                raise ValueError('tool result model/split/version differs from frozen study')
            artifact=Path(row['artifact_path'])
            fresh=result_row(artifact,planned[run_key(row)],rule['dataset_manifest'])
            if any(row.get(key)!=value for key,value in fresh.items()):
                raise ValueError('tool result differs from original run artifacts')
            raw=json.loads((artifact/'result.json').read_text())
            config=json.loads((artifact/'resolved_eval_config.json').read_text())
            tokenizer=json.loads((artifact/'tokenizer_manifest.json').read_text())
            profiles.add(canonical(dict(mode=config['mode'],budget=config['budget'],max_context_tokens=config['max_context_tokens'],max_output_tokens=config['max_output_tokens'],
                                        temperature=config.get('temperature',0),compression_trigger_tokens=config.get('compression_trigger_tokens',8192),
                                        chat_template_sha256=tokenizer['chat_template_sha256'],vocabulary_sha256=tokenizer['vocabulary_sha256'],template_kwargs=tokenizer['template_kwargs'])))
            source=Path(rule['dataset_manifest']).parent/entries[row['instance_id']]['path']
            task=json.loads((source/'task.json').read_text())['public_task']
            private=json.loads((source/'private_spec.json').read_text())
            grade=evaluate(artifact/'world.sqlite',{**task,**private},source/'initial_world.sqlite')
            audit=reconcile(artifact/'world.sqlite')
            if raw['termination']['state'] in ('INFRA_ERROR','PARSE_ERROR','BUDGET_STOP','CONTEXT_OVERFLOW') or raw.get('infrastructure_error') or not audit['consistent']:
                grade.update(full_success=False,reward=-1.0 if grade['hard_violations'] else 0.0)
            if grade!=raw['grade'] or audit!=raw['operation_audit']:
                raise ValueError('tool-equivalence independent regrading disagrees')
            versions.add(tuple(row[key] for key in VERSION_FIELDS));raw_results.append(raw)
            artifacts.append(dict(path=str(artifact),result_sha256=digest(artifact/'result.json'),world_sha256=digest(artifact/'world.sqlite'),trajectory_sha256=digest(artifact/'trajectory.jsonl')))
        sides.append(ordered);raw_sides.append(raw_results)
    if len(profiles)!=1:
        raise ValueError('tool-equivalence sampling/tokenizer profiles differ')
    if len(versions)!=1:
        raise ValueError('incomparable paired tool-result versions or budgets')
    comparison=compare_tools(*sides,rule['max_success_rate_delta'],rule['max_disagreement_rate'],rule['min_success_rate'])
    new_severe=any(set(right['grade']['hard_violations'])-set(left['grade']['hard_violations']) for left,right in zip(*raw_sides))
    audit_clean=all(raw['operation_audit']['consistent'] for side in raw_sides for raw in side)
    report=dict(passed=comparison['passed'] and not new_severe and audit_clean,checks=['fixed_next_token_logits','greedy_generation_tokens','paired_dev_tool_success','independent_regrading'],
                tool_comparison=comparison,new_severe_violation=new_severe,operation_audits_clean=audit_clean,study_sha256=digest(rule_path),numeric_report_sha256=digest(rule['numeric_report']),
                source_checkpoint_files_sha256=numeric['source_checkpoint_files_sha256'],exported_checkpoint_sha256=numeric['exported_checkpoint_sha256'],optimizer_step=numeric['optimizer_step'],artifacts=artifacts)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with Path(out).open('x') as stream:
        stream.write(canonical(report)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    first=sub.add_parser('freeze');first.add_argument('--spec',required=True);first.add_argument('--out',required=True)
    last=sub.add_parser('finalize');last.add_argument('--rule',required=True);last.add_argument('--out',required=True)
    args=parser.parse_args()
    result=freeze(json.loads(Path(args.spec).read_text()),args.out) if args.command=='freeze' else finalize(args.rule,args.out)
    print(json.dumps(result,indent=2))
    if result.get('passed') is False:
        raise SystemExit(1)

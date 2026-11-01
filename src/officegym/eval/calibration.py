"""Audit complete dev calibration evidence before a separate dataset freeze."""
import hashlib
import json
import math
from pathlib import Path
from .plan import missing_runs
from .export import result_row
from officegym.tasks.validate import validate_dataset
from officegym.harness.versions import HARNESS_VERSION


def audit_calibration(manifest_path,plan_path,results_path,out,*,min_success_rate,max_success_rate):
    for value in (min_success_rate,max_success_rate):
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite explicit calibration bounds required')
    if not 0<=min_success_rate<max_success_rate<=1:raise ValueError('ordered calibration bounds in [0,1] required')
    manifest_path,plan_path,results_path=map(lambda p:Path(p).resolve(),(manifest_path,plan_path,results_path))
    validate_dataset(manifest_path)
    manifest=json.loads(manifest_path.read_text());digest=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if Path(str(plan_path)+'.sha256').read_text().strip()!=digest:raise ValueError('calibration plan checksum mismatch')
    plan=[json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    rows=[json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    groups=('api_simple','api_full');dev={e['instance_id'] for e in manifest['instances'] if e['split']=='dev'}
    expected={(group,identity,repeat) for group in groups for identity in dev for repeat in (0,1,2)}
    if not expected or any(r['split']!='dev' for r in plan) or {(r['group'],r['instance_id'],r['repeat_id']) for r in plan}!=expected:
        raise ValueError('complete paired API dev plan required')
    if missing_runs(plan,rows):raise ValueError('calibration results incomplete')
    plans={(r['group'],r['split'],r['instance_id'],r['repeat_id']):r for r in plan}
    if any(row.get('harness_version')!=HARNESS_VERSION for row in rows):raise ValueError('current Harness calibration required')
    for field in ('model_version','budget_hash'):
        values={row.get(field) for row in rows}
        if len(values)!=1 or not all(isinstance(value,str) and value for value in values):raise ValueError('one shared calibration '+field+' required')
    artifacts=[]
    for row in rows:
        path=row.get('artifact_path')
        if not isinstance(path,str) or not path.strip():raise ValueError('original calibration artifact path required')
        artifacts.append(Path(path).resolve())
    if len(set(artifacts))!=len(rows):raise ValueError('calibration repeats cannot reuse runtime artifact directories')
    scores={group:[] for group in groups}
    for row in rows:
        if row.get('synthetic') is not False or row.get('example_only') is not False or row.get('run_plan_sha256')!=digest:
            raise ValueError('measured hash-bound calibration rows required')
        key=tuple(row[k] for k in ('group','split','instance_id','repeat_id'))
        actual=result_row(row['artifact_path'],plans[key],manifest_path)
        if any(row.get(k)!=v for k,v in actual.items()):raise ValueError('calibration row differs from original runtime evidence')
        scores[row['group']].append(actual['success'])
    rates={group:sum(values)/len(values) for group,values in scores.items()}
    useful=any(min_success_rate<rate<max_success_rate for rate in rates.values())
    report=dict(evidence_verified=True,calibration_usable=useful,success_rates=rates,dev_instances=len(dev),runs=len(rows),min_success_rate=min_success_rate,max_success_rate=max_success_rate,source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),plan_sha256=digest,results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),dataset_frozen=False,scope='Verified complete dev evidence and explicitly chosen difficulty bounds only; not dataset freeze or independent proof of pre-experiment chronology.')
    destination=Path(out);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report

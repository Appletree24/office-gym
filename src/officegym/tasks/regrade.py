"""Rescore retained development references without changing task inputs or old evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
from officegym.env.world import World,canonical
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
from .validate import validate_dataset,REQUIRED


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_hash(path):
    world=World.__new__(World)
    world.db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True);world.db.row_factory=sqlite3.Row
    try:return world.state_hash()
    finally:world.close()


def regrade(manifest_path,out,*,max_wall_seconds=180,repair_optional_material=False):
    if type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:raise ValueError('finite positive regrade wall budget required')
    if type(repair_optional_material) is not bool:raise ValueError('explicit annotation repair boolean required')
    metadata_changed=0
    started=time.monotonic();source=Path(manifest_path).resolve();original_sha=digest(source);manifest=json.loads(source.read_text())
    if manifest.get('status') not in ('development_dataset_not_frozen','development_distribution_not_frozen'):raise ValueError('only unfrozen development datasets may be regraded')
    root=Path(out).resolve()
    if root.is_relative_to(source.parent):raise ValueError('regrade output must be outside source dataset')
    root.mkdir(parents=True,exist_ok=False);records=[]
    with (root/'regrade_progress.jsonl').open('x') as progress:
        for entry in manifest['instances']:
            if time.monotonic()-started>=max_wall_seconds:raise TimeoutError('dataset regrade wall budget exhausted')
            directory=(source.parent/entry['path']).resolve();target=(root/entry['path']).resolve()
            if not directory.is_relative_to(source.parent) or not target.is_relative_to(root):raise ValueError('dataset path escapes regrade root')
            if any(file.is_symlink() for file in directory.rglob('*')):raise ValueError('regrade source contains symlinks')
            if any(file.stat().st_size for file in directory.glob('*-wal')):raise ValueError('checkpoint active source WAL before rescoring')
            expected=entry['files_sha256']
            if any(Path(name).name!=name or name in ('.','..') for name in expected):raise ValueError('invalid source inventory path')
            if not set(REQUIRED)<=set(expected) or any(digest(directory/name)!=value for name,value in expected.items()):raise ValueError('source dataset file hash mismatch')
            if final_hash(directory/'reference_final.sqlite')!=entry['reference_final_hash']:raise ValueError('retained reference final state changed')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(directory,target,ignore=shutil.ignore_patterns('*-wal','*-shm'))
            if any(digest(target/name)!=value for name,value in expected.items()):raise ValueError('copied dataset differs from source manifest')
            if repair_optional_material:
                from .dependencies import annotate_required_dependencies
                task=json.loads((target/'task.json').read_text());updated=annotate_required_dependencies(task)
                if canonical(updated)!=canonical(task):
                    metadata_changed+=1;(target/'task.json').write_text(canonical(updated)+'\n')
                    entry['files_sha256']['task.json']=digest(target/'task.json')
                entry['dependency_width']=updated['dependency_width']
            public=json.loads((target/'task.json').read_text())['public_task'];private=json.loads((target/'private_spec.json').read_text())
            grade=evaluate(target/'reference_final.sqlite',{**public,**private},target/'initial_world.sqlite');audit=reconcile(target/'reference_final.sqlite')
            if final_hash(target/'reference_final.sqlite')!=entry['reference_final_hash']:raise ValueError('reference state changed during regrade')
            (target/'reference_grade.json').write_text(canonical(grade)+'\n')
            entry['files_sha256']['reference_grade.json']=digest(target/'reference_grade.json')
            record=dict(instance_id=entry['instance_id'],full_success=grade['full_success'],operation_consistent=audit['consistent'],reference_final_hash=entry['reference_final_hash'])
            records.append(record);progress.write(canonical(record)+'\n');progress.flush()
    if digest(source)!=original_sha:raise ValueError('source manifest changed during regrade')
    if repair_optional_material:
        from .dependencies import GENERATOR_VERSION
        manifest['task_generator_version']=GENERATOR_VERSION
        manifest['limitations']=[item for item in manifest.get('limitations',[]) if not item.startswith('DAG width is 1 at controlled D1')]
        manifest['limitations'].append('Reply collection precedes invitation, not proposal/approval. Required width can be 2; controlled depth scans need matching-width redesign or stratification before freezing.')
    manifest['rescoring']=dict(source_manifest=str(source),source_manifest_sha256=original_sha,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,task_inputs_unchanged=not metadata_changed,public_business_inputs_unchanged=True,task_metadata_reannotated=metadata_changed)
    (root/'manifest.json').write_text(canonical(manifest)+'\n')
    report=dict(kind='development-reference-rescoring-not-model-results',instances=len(records),reference_passed=sum(row['full_success'] for row in records),operation_audits_passed=sum(row['operation_consistent'] for row in records),eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,source_manifest_sha256=original_sha,task_inputs_unchanged=not metadata_changed,public_business_inputs_unchanged=True,task_metadata_reannotated=metadata_changed,wall_cap_seconds=max_wall_seconds)
    (root/'regrade_report.json').write_text(canonical(report)+'\n')
    if not records or any(not row['full_success'] or not row['operation_consistent'] for row in records):raise ValueError('regraded reference failed; inspect retained results')
    validation=validate_dataset(root/'manifest.json');(root/'validation.json').write_text(canonical(validation)+'\n')
    report['elapsed_seconds']=time.monotonic()-started
    (root/'regrade_report.json').write_text(canonical(report)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True);parser.add_argument('--max-wall-seconds',type=float,default=180);parser.add_argument('--repair-dependencies','--repair-optional-observations','--repair-optional-material',dest='repair_optional_material',action='store_true');args=parser.parse_args()
    print(json.dumps(regrade(args.manifest,args.out,max_wall_seconds=args.max_wall_seconds,repair_optional_material=args.repair_optional_material),indent=2))

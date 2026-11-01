"""Content-bound dataset admission after independently re-audited dev calibration."""
import hashlib
import json
from pathlib import Path
import tempfile
from officegym.eval.calibration import audit_calibration
from officegym.eval.plan import freeze_plan
from officegym.graders.versions import EVAL_GRADER_VERSION, REWARD_VERSION
from officegym.harness.versions import HARNESS_VERSION
from officegym.tools.schema import TOOL_SCHEMA_VERSION
from .validate import validate_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_inventory():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.sql')}


def bound_files(manifest,plan,results):
    data=json.loads(manifest.read_text())
    paths={manifest,plan,Path(str(plan)+'.sha256'),results}
    for entry in data['instances']:
        paths.update((manifest.parent/entry['path']/name).resolve() for name in entry['files_sha256'])
    for line in results.read_text().splitlines():
        if not line.strip():continue
        root=Path(json.loads(line)['artifact_path']).resolve()
        for path in root.rglob('*'):
            if path.is_symlink():raise ValueError('calibration artifact symlinks cannot be frozen')
            if path.is_file():paths.add(path.resolve())
    return {str(path):digest(path) for path in sorted(paths)}


def freeze_dataset(manifest_path,grader,plan_path,results_path,out,*,min_success_rate,max_success_rate):
    if grader!=EVAL_GRADER_VERSION:raise ValueError('freeze requires the current exact eval grader version')
    manifest,plan,results=map(lambda p:Path(p).resolve(),(manifest_path,plan_path,results_path))
    # Hash before the audit as well as after it; an audit report supplied by a
    # caller cannot stand in for verification of original runtime evidence.
    before=bound_files(manifest,plan,results);code=code_inventory()
    with tempfile.TemporaryDirectory(prefix='officegym-freeze-audit-') as temporary:
        report=audit_calibration(manifest,plan,results,Path(temporary)/'audit.json',
                                 min_success_rate=min_success_rate,max_success_rate=max_success_rate)
    if report.get('evidence_verified') is not True or report.get('calibration_usable') is not True:
        raise ValueError('usable verified dev calibration required before dataset freeze')
    if before!=bound_files(manifest,plan,results) or code!=code_inventory():
        raise ValueError('freeze inputs or implementation changed during admission')
    record=dict(freeze_schema_version='dataset-freeze-v1',dataset_frozen=True,
                manifest_path=str(manifest),manifest_sha256=digest(manifest),
                calibration_plan_path=str(plan),calibration_results_path=str(results),
                calibration=report,bound_files_sha256=before,implementation_sha256=code,
                versions=dict(eval_grader=grader,reward=REWARD_VERSION,harness=HARNESS_VERSION,tools=TOOL_SCHEMA_VERSION),
                scope='Content-bound admission receipt; source files are not made immutable. Does not independently establish pre-experiment chronology or model-training acceptance.')
    freeze_plan([record],out)
    return record


def verify_frozen_dataset(path,manifest_path=None):
    path=Path(path).resolve()
    if Path(str(path)+'.sha256').read_text().strip()!=digest(path):raise ValueError('dataset freeze checksum mismatch')
    record=json.loads(path.read_text())
    if record.get('freeze_schema_version')!='dataset-freeze-v1' or record.get('dataset_frozen') is not True:
        raise ValueError('unsupported dataset freeze receipt')
    manifest=Path(record['manifest_path']).resolve()
    if manifest_path is not None and manifest!=Path(manifest_path).resolve():raise ValueError('freeze belongs to a different dataset manifest')
    expected=dict(eval_grader=EVAL_GRADER_VERSION,reward=REWARD_VERSION,harness=HARNESS_VERSION,tools=TOOL_SCHEMA_VERSION)
    if record.get('versions')!=expected or record.get('implementation_sha256')!=code_inventory():raise ValueError('frozen implementation or versions changed')
    report=record['calibration']
    if report.get('evidence_verified') is not True or report.get('calibration_usable') is not True:raise ValueError('freeze lacks admitted calibration')
    actual=bound_files(manifest,Path(record['calibration_plan_path']),Path(record['calibration_results_path']))
    if actual!=record['bound_files_sha256'] or digest(manifest)!=record['manifest_sha256']:raise ValueError('frozen dataset or calibration artifacts changed')
    validate_dataset(manifest)
    return record

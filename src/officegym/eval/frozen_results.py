"""Re-export original runtime evidence before reporting a frozen comparison."""
import json
from pathlib import Path
from officegym.tasks.freeze import verify_frozen_dataset,digest
from .export import result_row
from .plan import missing_runs,run_key


def audit_frozen_results(rows,plan_path,freeze_path):
    if not rows:raise ValueError('nonempty frozen comparison required')
    freeze_path,plan_path=map(lambda p:Path(p).resolve(),(freeze_path,plan_path))
    receipt_digest=digest(freeze_path)
    record=verify_frozen_dataset(freeze_path)
    plan_digest=digest(plan_path)
    if Path(str(plan_path)+'.sha256').read_text().strip()!=plan_digest:raise ValueError('frozen comparison plan checksum mismatch')
    plan=[json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    # Partial selected comparisons are checked for completeness by the paired
    # analysis; this check rejects duplicate/unplanned selected rows as well.
    missing_runs(plan,rows)
    planned={run_key(row):row for row in plan};paths=set()
    for row in rows:
        if row.get('dataset_freeze_sha256')!=receipt_digest or Path(row.get('dataset_freeze_path','')).resolve()!=freeze_path:
            raise ValueError('comparison row lacks matching dataset freeze admission')
        if row.get('run_plan_sha256')!=plan_digest:raise ValueError('comparison row differs from frozen run plan')
        raw=row.get('artifact_path')
        if not isinstance(raw,str) or not raw.strip():raise ValueError('original frozen comparison artifact path required')
        path=Path(raw).resolve()
        if path in paths:raise ValueError('frozen comparison repeats reuse runtime artifacts')
        paths.add(path)
        selection=json.loads((path.parent/'selection.json').read_text())
        if selection.get('plan_sha256')!=plan_digest:raise ValueError('runtime batch differs from frozen run plan')
        actual=result_row(path,planned[run_key(row)],record['manifest_path'])
        if any(row.get(key)!=value for key,value in actual.items()):raise ValueError('frozen comparison row differs from original runtime evidence')
    if digest(freeze_path)!=receipt_digest or digest(plan_path)!=plan_digest:raise ValueError('frozen comparison inputs changed during audit')
    return dict(dataset_freeze_sha256=receipt_digest,dataset_manifest_sha256=record['manifest_sha256'],run_plan_sha256=plan_digest,runtime_rows_verified=len(rows),scope='Content-bound original-runtime audit; not independent proof of pre-experiment chronology.')

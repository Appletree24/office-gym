"""Re-execute reference actions from immutable initial task snapshots."""
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate
from .reference import solve
from .validate import validate_dataset


def reference_manifest(manifest_path,out):
    manifest_path = Path(manifest_path).resolve()
    validate_dataset(manifest_path)
    entries = json.loads(manifest_path.read_text())['instances']
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    results = []
    for entry in entries:
        source = manifest_path.parent/entry['path']
        public = json.loads((source/'task.json').read_text())['public_task']
        private = json.loads((source/'private_spec.json').read_text())
        directory = out/entry['instance_id']
        directory.mkdir()
        world = clone_world(source/'initial_world.sqlite',directory/'final.sqlite')
        try:
            client = Dispatcher(world)
            with (directory/'trace.jsonl').open('w') as trace:
                def call(tool,args,operation=None):
                    result = client.call(tool,args,operation)
                    trace.write(canonical(dict(tool=tool,arguments=args,operation_id=operation,result=result))+'\n')
                    return result
                solve(public,call)
            grade = evaluate(directory/'final.sqlite',{**public,**private},source/'initial_world.sqlite')
            fingerprint = world.state_hash()
            record = dict(instance_id=entry['instance_id'],grade=grade,final_hash=fingerprint,replay_matches=fingerprint==entry['reference_final_hash'])
            (directory/'grade.json').write_text(canonical(record)+'\n')
            results.append(record)
        finally:
            world.close()
    summary = dict(instances=len(results),reference_passed=sum(row['grade']['full_success'] for row in results),replay_matched=sum(row['replay_matches'] for row in results),kind='reference_software_validation_not_model_results')
    (out/'summary.json').write_text(canonical(summary)+'\n')
    if any(not row['grade']['full_success'] or not row['replay_matches'] for row in results):
        raise ValueError('reference or deterministic replay failed; inspect saved records')
    return summary

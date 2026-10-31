"""Prepare train-only JSONL for the pinned verl RLHFDataset loader."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.harness.loop import initial_messages
from officegym.tasks.validate import validate_dataset
from .worker_plugins import write_plugins
from .rollout_contract import bind_rollout_rows,validate_rollout_contract


def prepare(manifest_path,contract,out):
    manifest_path = Path(manifest_path).resolve()
    validate_dataset(manifest_path)
    validate_rollout_contract(contract,contract['global_rollout_token_limit'],contract['actor_version'])
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for entry in manifest['instances']:
        if entry['split']!='train':
            continue
        source = (manifest_path.parent/entry['path']).resolve()
        task = json.loads((source/'task.json').read_text())
        rows.append(dict(agent_name='officegym',data_source='officegym',
                         prompt=initial_messages(task['public_task'],full=True),
                         officegym=dict(instance_dir=str(source)),
                         extra_info=dict(index=entry['instance_id']),
                         reward_model=dict(style='rule',ground_truth='officegym_independent_grader')))
    rows = bind_rollout_rows(rows,contract)
    root = Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    data = ''.join(canonical(row)+'\n' for row in rows).encode()
    (root/'train.jsonl').write_bytes(data)
    plugins = write_plugins(root)
    report = dict(format='officegym_verl_train_rows_v1',instances=len(rows),split='train',
                  dataset_manifest=str(manifest_path),dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  train_file=str(root/'train.jsonl'),train_sha256=hashlib.sha256(data).hexdigest(),
                  worker_plugins=plugins,rollout_contract=contract,status='prepared_not_backend_runtime_verified')
    (root/'preparation.json').write_text(canonical(report)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest,json.loads(Path(args.contract).read_text()),args.out),indent=2))

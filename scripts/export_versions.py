"""Export reproducibility metadata without inspecting credentials or environment values."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def git(root,*args):
    result = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,check=False)
    return result.stdout.strip() if result.returncode==0 else None


def export(root):
    root = Path(root).resolve()
    groups = {
        'implementation':sorted(root.joinpath('src').rglob('*.py')),
        'schema':sorted(root.joinpath('schema').glob('*.sql')),
        'contracts':sorted(root.joinpath('contracts').glob('*.json')),
        'configuration':sorted(path for path in root.joinpath('configs').rglob('*') if path.is_file() and path.suffix in ('.json','.yaml','.yml','.toml')),
        'dependency_locks':sorted(root.glob('requirements*.lock')),
        'entrypoints':sorted(path for path in root.joinpath('scripts').rglob('*') if path.is_file() and path.suffix in ('.py','.sh')),
        'packaging':[root/'pyproject.toml'] if (root/'pyproject.toml').is_file() else [],
    }
    files=sorted(set(path for paths in groups.values() for path in paths))
    hashes = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    inventory = {kind:[str(path.relative_to(root)) for path in paths] for kind,paths in groups.items()}
    versions = {}
    for name in ('mcp','torch','transformers','trl','peft','verl','vllm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = git(root,'status','--porcelain')
    return {'kind':'software_version_manifest_not_experiment','python':platform.python_version(),
            'git_commit':git(root,'rev-parse','HEAD'),'git_dirty':bool(status) if status is not None else None,
            'packages':versions,'file_sha256':hashes,'inventory_version':'reproducibility-source-inventory-v2','inventory_groups':inventory,
            'scope':'Repository source/configuration/lock inventory and installed package versions only; not a resolved runtime, model artifact inventory, training admission or scientific completion.',
            'source_manifest_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(export(args.root),indent=2)+'\n')

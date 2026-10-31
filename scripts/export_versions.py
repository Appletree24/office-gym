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
    files = sorted([*root.joinpath('src').rglob('*.py'),*root.joinpath('schema').glob('*.sql'),*root.joinpath('contracts').glob('*.json')])
    hashes = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    versions = {}
    for name in ('mcp','torch','transformers','trl','peft','verl','vllm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = git(root,'status','--porcelain')
    return {'kind':'software_version_manifest_not_experiment','python':platform.python_version(),
            'git_commit':git(root,'rev-parse','HEAD'),'git_dirty':bool(status) if status is not None else None,
            'packages':versions,'file_sha256':hashes,
            'source_manifest_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(export(args.root),indent=2)+'\n')

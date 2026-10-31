"""Run provenance without credential/environment-value capture."""
import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from officegym.env.world import canonical
from officegym.tools.schema import tool_schemas,CONTRACT
from officegym.tools.output_schema import TOOL_RESULT_SCHEMA


def provenance(instance_dir,policy):
    package_root = Path(__file__).resolve().parents[1]
    root = package_root.parent.parent
    source_checkout = (root/'.git').exists() and (root/'src'/'officegym').resolve()==package_root
    def git(*arguments):
        if not source_checkout:
            return None
        result = subprocess.run(['git','-C',str(root),*arguments],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    packages = {}
    for name in ('mcp','transformers','torch','trl','peft','verl','vllm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = {str(path.relative_to(package_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package_root.rglob('*')) if path.is_file() and path.suffix in ('.py','.sql','.json')}
    artifacts = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(instance_dir).iterdir() if path.name in ('task.json','private_spec.json','event_rules.json','initial_world.sqlite')}
    categories = {category:sum(tool['category']==category for tool in CONTRACT['tools']) for category in ('read','write','control')}
    return {'git_commit':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')) if source_checkout else None,'source_kind':'checkout' if source_checkout else 'installed_distribution','source_sha256':hashlib.sha256(canonical(files).encode()).hexdigest(),
            'instance_artifact_sha256':artifacts,'packages':packages,'model_version':getattr(policy,'model_version',None),
            'tool_schema_sha256':hashlib.sha256(canonical({'input':tool_schemas(),'output':TOOL_RESULT_SCHEMA}).encode()).hexdigest(),
            'tool_counts':categories,'env_version':'env-v1','tool_schema_version':'tools-v1','harness_version':'harness-v1-development',
            'reward_version':'reward-v1','eval_grader_version':'eval-v1-development'}

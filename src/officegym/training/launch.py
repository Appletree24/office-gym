"""Launch only an explicitly resolved and compatibility-tested upstream command."""
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from officegym.env.world import canonical
from .rollout_contract import validate_rollout_contract
from .verl_entry import bound_command,load_resolved
from .worker_plugins import validate_plugins

VERL_COMMIT = '8df5baeed0f5ecdd7f870d83bc3d79194816e2a9'
SMOKE_CHECKS = {'isolated_worlds','native_token_ids','tool_loss_mask_zero','initial_actor_reference_kl','parameter_update','actor_version_sync','checkpoint_export_evaluation'}


def compatibility_hash(resolved):
    # The sole permitted smoke-to-training change is the number of updates.
    compatible = json.loads(canonical(resolved))
    compatible.get('trainer',{}).pop('total_training_steps',None)
    return hashlib.sha256(canonical(compatible).encode()).hexdigest()


def validate_launch(config,smoke=False):
    required = {'verl_commit','verl_checkout','locked_packages','resolved_upstream_config','resolved_upstream_sha256','command','max_updates','max_wall_seconds','max_rollout_tokens','starting_checkpoint_hash','reference_checkpoint_hash','group_size','rollout_contract','prepared_rollouts'}
    if required-set(config):
        raise ValueError('missing locked launch fields: '+','.join(sorted(required-set(config))))
    if config['verl_commit']!=VERL_COMMIT:
        raise ValueError('verl revision differs from inspected adapter source')
    for key in ('max_updates','max_rollout_tokens'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError('positive finite budget required: '+key)
    if type(config['max_wall_seconds']) not in (int,float) or not math.isfinite(config['max_wall_seconds']) or config['max_wall_seconds']<=0:
        raise ValueError('finite training wall budget required')
    if type(config['group_size']) is not int or config['group_size']<2:
        raise ValueError('GRPO needs at least two rollouts per task')
    if smoke and config['max_updates']>5:
        raise ValueError('smoke launch limited to five updates')
    if config['starting_checkpoint_hash']!=config['reference_checkpoint_hash'] or not config['starting_checkpoint_hash'] or config['starting_checkpoint_hash'].startswith('SET_'):
        raise ValueError('actor and reference must start from identical exported SFT')
    validate_rollout_contract(config['rollout_contract'],config['max_rollout_tokens'],config['starting_checkpoint_hash'])
    command = config['command']
    if not isinstance(command,list) or not command or any(not isinstance(arg,str) or not arg for arg in command):
        raise ValueError('explicit argv list required')
    upstream = Path(config['resolved_upstream_config'])
    if not upstream.is_file() or hashlib.sha256(upstream.read_bytes()).hexdigest()!=config['resolved_upstream_sha256']:
        raise ValueError('resolved upstream configuration hash mismatch')
    resolved = load_resolved(upstream,config['resolved_upstream_sha256'])
    values = {}
    def flatten(mapping,prefix=''):
        for key,value in mapping.items():
            name = prefix+str(key)
            if isinstance(value,dict):
                flatten(value,name+'.')
            else:
                values[name] = value
    flatten(resolved)
    required_values = {'algorithm.adv_estimator':'grpo','actor_rollout_ref.rollout.n':config['group_size'],'trainer.total_training_steps':config['max_updates']}
    if any(values.get(key)!=value for key,value in required_values.items()):
        raise ValueError('resolved upstream algorithm/group/update budget mismatch')
    preparation = json.loads(Path(config['prepared_rollouts']).read_text())
    train_file = Path(preparation['train_file'])
    if preparation.get('rollout_contract')!=config['rollout_contract'] or preparation.get('split')!='train':
        raise ValueError('prepared rollout contract or split mismatch')
    if not train_file.is_absolute() or hashlib.sha256(train_file.read_bytes()).hexdigest()!=preparation['train_sha256']:
        raise ValueError('prepared rollout data hash mismatch')
    validate_plugins(preparation.get('worker_plugins',{}),values)
    train_files = values.get('data.train_files')
    if train_files!=str(train_file) and train_files!=[str(train_file)]:
        raise ValueError('upstream train files differ from bound rollout data')
    if not smoke:
        report = json.loads(Path(config['smoke_report']).read_text())
        if report.get('passed') is not True or not SMOKE_CHECKS <= set(report.get('passed_checks',[])):
            raise ValueError('complete rollout/update/export smoke evidence required')
        if report.get('compatibility_sha256')!=compatibility_hash(resolved):
            raise ValueError('training parameters differ from verified smoke config')
    if command!=bound_command(upstream,config['resolved_upstream_sha256']):
        raise ValueError('command must execute the exact hashed upstream configuration')
    return command


def execute_bounded(command,stdout,stderr,max_wall_seconds):
    process = subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
    exhausted = False
    try:
        process.wait(timeout=max_wall_seconds)
    except subprocess.TimeoutExpired:
        exhausted = True
    finally:
        # Also clean workers that outlive a normally exited launcher.
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return {'exit_code':process.returncode,'wall_budget_exhausted':exhausted,'process_group_cleanup':True}


def launch(config_path,out,smoke=False):
    config = json.loads(Path(config_path).read_text())
    command = validate_launch(config,smoke)
    for package in ('torch','transformers','verl','vllm'):
        if importlib.metadata.version(package)!=config['locked_packages'].get(package):
            raise ValueError('runtime dependency differs from lock: '+package)
    commit = subprocess.run(['git','-C',config['verl_checkout'],'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if commit!=config['verl_commit']:
        raise ValueError('actual verl checkout differs from lock')
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    (root/'launch_config.json').write_text(canonical(config)+'\n')
    resolved = load_resolved(config['resolved_upstream_config'],config['resolved_upstream_sha256'])
    (root/'resolved_upstream.json').write_text(canonical(resolved)+'\n')
    started = time.monotonic()
    with (root/'upstream.stdout.log').open('w') as stdout, (root/'upstream.stderr.log').open('w') as stderr:
        status = execute_bounded(command,stdout,stderr,config['max_wall_seconds'])
    status.update(compatibility_sha256=compatibility_hash(resolved),resolved_upstream_sha256=config['resolved_upstream_sha256'],wall_seconds=time.monotonic()-started,smoke=smoke,semantic_smoke_verification='pending')
    (root/'launch_result.json').write_text(canonical(status)+'\n')
    return status


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    print(json.dumps(launch(args.config,args.out,args.smoke),indent=2))

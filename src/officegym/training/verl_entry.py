"""Load exactly one hashed resolved config; accept no Hydra overrides."""
import hashlib
import json
import sys
from pathlib import Path


def load_resolved(path,sha256):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=sha256:
        raise ValueError('upstream configuration changed before execution')
    try:
        config=json.loads(payload)
    except ValueError:
        import yaml
        config=yaml.safe_load(payload)
    def check(value):
        if isinstance(value,str) and '${' in value:
            raise ValueError('upstream configuration must have resolved interpolations')
        if isinstance(value,dict):
            for child in value.values():
                check(child)
        elif isinstance(value,list):
            for child in value:
                check(child)
    if not isinstance(config,dict) or 'defaults' in config:
        raise ValueError('fully resolved upstream mapping required')
    check(config)
    return config


def bound_command(path,sha256):
    return [sys.executable,'-m','officegym.training.verl_entry','--config',str(Path(path).resolve()),'--sha256',sha256]


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    resolved=load_resolved(args.config,args.sha256)
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import main as upstream_main
    # Pinned main's unwrapped function retains device/config validation and
    # task-runner selection, without composing defaults or accepting overrides.
    upstream_main.__wrapped__(OmegaConf.create(resolved))


if __name__=='__main__':
    main()

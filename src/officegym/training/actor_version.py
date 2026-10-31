"""Fail-closed logical actor identity and per-generation backend version checks.

Step tags identify the admitted branch and completed updates; they are not a
replacement for measuring physical actor/rollout weight equivalence.
"""
from pathlib import Path
from officegym.eval.artifacts import write_json


def actor_identity(initial,step):
    if not isinstance(initial,str) or not initial or type(step) is not int or step<0:
        raise ValueError('initial actor and nonnegative integer update step required')
    return initial if step==0 else initial+':optimizer_step:'+str(step)


def bind_actor_metadata(metadata,initial,step):
    version=actor_identity(initial,step)
    if not isinstance(metadata,dict) or metadata.get('actor_version')!=initial:
        raise ValueError('prepared rollout actor differs from admitted starting actor')
    if {'starting_actor_version','actor_update_step'} & metadata.keys():
        raise ValueError('prepared rollout overrides controller actor version')
    return {**metadata,'starting_actor_version':initial,'actor_update_step':step,'actor_version':version}


def validate_actor_metadata(metadata,initial=None):
    starting=metadata.get('starting_actor_version');step=metadata.get('actor_update_step')
    expected=actor_identity(starting,step)
    if metadata.get('actor_version')!=expected or (initial is not None and starting!=initial):
        raise ValueError('rollout actor version does not match trusted branch/update')
    return step


class VersionedServer:
    def __init__(self,server,step,artifact_dir):
        actor_identity('validation',step)
        self.server=server;self.step=step;self.artifact_dir=Path(artifact_dir);self.checks=0

    def __getattr__(self,name):return getattr(self.server,name)

    async def generate(self,*args,**kwargs):
        output=await self.server.generate(*args,**kwargs)
        fields=output.extra_fields
        valid=all(type(fields.get(key)) is int and fields[key]==self.step for key in ('min_global_steps','max_global_steps'))
        if 'global_steps' in fields:
            valid=valid and type(fields['global_steps']) is int and fields['global_steps']==self.step
        self.checks+=1
        write_json(self.artifact_dir/f'generation_version_{self.checks:04d}.json',dict(expected_update_step=self.step,backend_versions={key:fields.get(key) for key in ('global_steps','min_global_steps','max_global_steps')},accepted=valid,original_token_ids=list(output.token_ids),original_logprobs=output.log_probs))
        if not valid:raise ValueError('backend generation used missing, stale or mixed actor versions')
        return output


def validate_actor_batch(columns,initial,step,*,expected_rows=None):
    expected=actor_identity(initial,step)
    names=('officegym_actor_version','officegym_actor_update_step','officegym_actor_version_verified','min_global_steps','max_global_steps')
    if any(name not in columns for name in names):raise ValueError('native optimizer batch lacks actor version evidence')
    lengths={len(columns[name]) for name in names}
    if len(lengths)!=1 or not next(iter(lengths)):raise ValueError('native actor version evidence lengths differ or empty')
    if expected_rows is not None and next(iter(lengths))!=expected_rows:raise ValueError('native actor version evidence does not cover every optimizer row')
    for version,update,verified,minimum,maximum in zip(*(columns[name] for name in names)):
        if version!=expected or verified is not True or any(type(value) is not int or value!=step for value in (update,minimum,maximum)):
            raise ValueError('native optimizer batch contains stale or mixed actor versions')

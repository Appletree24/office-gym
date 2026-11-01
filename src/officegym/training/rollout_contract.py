"""Validate trusted row metadata against one launch-wide resource contract."""
from pathlib import Path


def parse_retry_budget(contract):
    value=contract.get('parse_retries',2)
    if type(value) is not int or value<0:raise ValueError('finite nonnegative parse retry budget required')
    return value


def validate_rollout_contract(contract, max_rollout_tokens, actor_version):
    required = {'token_ledger_path','global_rollout_token_limit','max_generation_tokens',
                'max_context_tokens','max_tool_calls','actor_version','artifact_root'}
    if not isinstance(contract,dict) or required-set(contract):
        raise ValueError('complete trusted rollout resource contract required')
    for name in ('global_rollout_token_limit','max_generation_tokens','max_context_tokens','max_tool_calls'):
        if type(contract[name]) is not int or contract[name]<1:
            raise ValueError('positive finite rollout resource required: '+name)
    if contract['max_context_tokens']<2 or contract['max_generation_tokens']>max_rollout_tokens:
        raise ValueError('rollout reservation exceeds global budget or invalid context')
    if contract['global_rollout_token_limit']!=max_rollout_tokens:
        raise ValueError('rollout global token limit differs from launch budget')
    if contract['actor_version']!=actor_version:
        raise ValueError('rollout initial actor differs from launch checkpoint')
    for name in ('token_ledger_path','artifact_root'):
        if not isinstance(contract[name],str) or not Path(contract[name]).is_absolute():
            raise ValueError('shared absolute rollout path required: '+name)
    return dict(contract,parse_retries=parse_retry_budget(contract))


def bind_rollout_rows(rows,contract):
    """Copy rows, reject overrides and attach contract before backend serialization.

    Training updates must replace actor_version with the actual synchronized actor
    version before dispatch; the initial contract binds the starting checkpoint.
    """
    contract=dict(contract,parse_retries=parse_retry_budget(contract))
    bound = []
    for row in rows:
        metadata = row.get('officegym')
        if not isinstance(metadata,dict) or not metadata.get('instance_dir'):
            raise ValueError('trusted instance_dir required in every rollout row')
        conflicts = [key for key,value in contract.items() if key in metadata and metadata[key]!=value]
        if conflicts:
            raise ValueError('rollout row overrides launch contract: '+','.join(sorted(conflicts)))
        bound.append({**row,'officegym':{**metadata,**contract}})
    if not bound:
        raise ValueError('nonempty rollout rows required')
    return bound

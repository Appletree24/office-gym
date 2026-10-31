"""Bounded effective-group sampling; optimizer/backend integration is separate."""
from collections import Counter
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rollout:
    trajectory_id: str
    instance_id: str
    actor_version: str
    reward: float
    generated_tokens: int


def collect_groups(sample_group, *, actor_version, target_groups, group_size=4, max_multiplier=4, token_budget):
    """sample_group(version, size, remaining_tokens) must enforce its generation cap."""
    if any(type(value) is not int or value <= 0 for value in (target_groups,group_size,max_multiplier,token_budget)) or group_size < 2:
        raise ValueError('invalid finite sampling budget')
    if not actor_version:
        raise ValueError('actor version required')
    effective, seen = [], set()
    candidates = tokens = 0
    histogram = Counter()
    while len(effective)<target_groups and candidates<target_groups*max_multiplier and tokens<token_budget:
        group = list(sample_group(actor_version,group_size,token_budget-tokens))
        candidates += 1
        if len(group) != group_size:
            raise ValueError('incomplete rollout group')
        if any(not isinstance(row,Rollout) for row in group):
            raise ValueError('invalid rollout record')
        if len({row.instance_id for row in group}) != 1:
            raise ValueError('group must share one initial task')
        for row in group:
            if row.actor_version != actor_version:
                raise ValueError('actor changed during update refill')
            if not row.trajectory_id or row.trajectory_id in seen:
                raise ValueError('reused trajectory')
            if not math.isfinite(row.reward) or row.reward not in (-1,0,1):
                raise ValueError('reward-v1 requires terminal 1/0/-1')
            if type(row.generated_tokens) is not int or row.generated_tokens<0:
                raise ValueError('invalid token usage')
            seen.add(row.trajectory_id)
            tokens += row.generated_tokens
            histogram[str(row.reward)] += 1
        if tokens>token_budget:
            raise ValueError('rollout backend exceeded finite token budget')
        if len({row.reward for row in group})>1:
            effective.append(group)
    ready = len(effective)==target_groups
    return {'groups':effective,'should_update':ready,'skip_reason':None if ready else 'insufficient_effective_groups',
            'actor_version':actor_version,'candidate_groups':candidates,'effective_groups':len(effective),
            'effective_group_ratio':len(effective)/candidates if candidates else 0,
            'rollout_tokens':tokens,'refill_attempts':max(0,candidates-target_groups),'reward_histogram':dict(histogram)}


def relative_advantages(rewards):
    if len(rewards)<2 or any(not math.isfinite(value) for value in rewards):
        raise ValueError('invalid rewards')
    mean = sum(rewards)/len(rewards)
    variance = sum((value-mean)**2 for value in rewards)/len(rewards)
    if variance==0:
        return [0.0]*len(rewards)
    return [(value-mean)/math.sqrt(variance) for value in rewards]

"""Training trajectory gates independent of a particular tokenizer framework."""
from officegym.graders.versions import require_current_scoring
import hashlib
from officegym.env.world import canonical


class RejectedTrajectory(ValueError):
    pass


def build_sample(trajectory, tokenize, max_length=8192):
    """tokenize(messages, tools) must use the locked real chat template.

    It returns input_ids, assistant_mask and token_roles aligned to each original
    rendered token. No decoding/re-encoding, truncation or synthetic masks here.
    """
    if trajectory.get('split') != 'train':
        raise RejectedTrajectory('non_training_split')
    if not trajectory.get('grade',{}).get('full_success'):
        raise RejectedTrajectory('not_verified_success')
    try:
        require_current_scoring(trajectory['grade'])
    except ValueError as exc:
        raise RejectedTrajectory('stale_scoring_version') from exc
    if trajectory.get('compacted') or trajectory.get('recovered'):
        raise RejectedTrajectory('compressed_or_recovered')
    if not trajectory.get('instance_id') or not trajectory.get('messages'):
        raise RejectedTrajectory('missing_provenance_or_messages')
    encoded = tokenize(trajectory['messages'],trajectory['tools'])
    ids, mask, roles = encoded['input_ids'], encoded['assistant_mask'], encoded['token_roles']
    verify_mask(ids,mask,roles)
    if len(ids) > max_length:
        raise RejectedTrajectory('context_overflow')
    return {'instance_id':trajectory['instance_id'],'input_ids':list(ids),
            'labels':[token if bit else -100 for token,bit in zip(ids,mask)],
            'assistant_mask':list(mask),'token_roles':list(roles),
            'trajectory_sha256':hashlib.sha256(canonical(trajectory).encode()).hexdigest()}


def verify_mask(ids,mask,roles):
    if not ids or not len(ids)==len(mask)==len(roles):
        raise RejectedTrajectory('unaligned_tokens')
    if any(type(token) is not int or token<0 for token in ids):
        raise RejectedTrajectory('invalid_token_ids')
    if any(type(bit) is not int or bit not in (0,1) for bit in mask):
        raise RejectedTrajectory('invalid_loss_mask')
    if not any(mask):
        raise RejectedTrajectory('empty_assistant_mask')
    for role,bit in zip(roles,mask):
        if role not in ('system','user','assistant','tool','template'):
            raise RejectedTrajectory('unknown_token_role')
        if bit and role != 'assistant':
            raise RejectedTrajectory('non_actor_token_has_loss')


def validate_rollout(ids,response_mask,roles,logprobs):
    """Validate original backend token IDs; caller retains them without recoding."""
    verify_mask(ids,response_mask,roles)
    if len(logprobs) != len(ids):
        raise RejectedTrajectory('unaligned_logprobs')
    import math
    if any(not isinstance(value,(int,float)) or not math.isfinite(value) for value in logprobs):
        raise RejectedTrajectory('invalid_logprobs')

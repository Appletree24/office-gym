"""Shared grader/reward identities; evidence fixes invalidate prior scoring."""
EVAL_GRADER_VERSION = 'eval-v12-historical-materials'
REWARD_VERSION = 'reward-v12-historical-materials'


def require_current_scoring(record):
    if record.get('eval_grader_version')!=EVAL_GRADER_VERSION or record.get('reward_version')!=REWARD_VERSION:
        raise ValueError('stale or missing scoring version; regrade original world before reuse')

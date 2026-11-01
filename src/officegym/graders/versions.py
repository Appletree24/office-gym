"""Shared grader/reward identities; evidence fixes invalidate prior scoring."""
EVAL_GRADER_VERSION = 'eval-v7-meeting-creation-scope'
REWARD_VERSION = 'reward-v7-meeting-creation-scope'


def require_current_scoring(record):
    if record.get('eval_grader_version')!=EVAL_GRADER_VERSION or record.get('reward_version')!=REWARD_VERSION:
        raise ValueError('stale or missing scoring version; regrade original world before reuse')

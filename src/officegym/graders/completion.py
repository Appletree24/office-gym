"""Completion evidence separates world-state checks from trusted runtime stops."""
GRADE_SCHEMA_VERSION='grade-result-v2-completion-evidence'


def world_completion(*,outcome_ok,hard_violations,soft_deviations,unexpected_mutations,within_budget,declared_done):
    return dict(outcome_ok=outcome_ok,no_hard_violations=not hard_violations,no_soft_deviations=not soft_deviations,no_unexpected_mutations=not unexpected_mutations,within_simulated_deadline=within_budget,declared_done=declared_done)


def record_runtime_termination(grade,*,reason,source,accepted):
    if not isinstance(reason,str) or not reason or source not in ('harness_runtime','native_rollout') or type(accepted) is not bool:
        raise ValueError('explicit trusted runtime termination evidence required')
    result=dict(grade);checks=dict(grade['completion_checks']);checks['runtime_accepted']=accepted
    result.update(world_terminated_reason=grade.get('world_terminated_reason',grade['terminated_reason']),terminated_reason=reason,termination_reason_source=source,completion_checks=checks)
    return result

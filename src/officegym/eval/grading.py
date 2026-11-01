"""Shared trusted episode grading for execution and offline export verification."""
from officegym.graders.result import evaluate
from officegym.graders.operations import reconcile
from officegym.graders.completion import record_runtime_termination

TERMINAL_STATES=frozenset(('DECLARED_DONE','INFRA_ERROR','PARSE_ERROR','BUDGET_STOP','CONTEXT_OVERFLOW'))


def grade_episode_snapshot(world_path,spec,initial_path,*,termination_state,infrastructure_error=None):
    if termination_state not in TERMINAL_STATES:raise ValueError('known terminal Harness state required')
    grade=evaluate(world_path,spec,initial_path);audit=reconcile(world_path)
    rejected=termination_state!='DECLARED_DONE' or bool(infrastructure_error) or not audit['consistent']
    if rejected:
        grade['full_success']=False
        grade['reward']=-1.0 if grade['hard_violations'] else 0.0
    grade=record_runtime_termination(grade,reason=termination_state,source='harness_runtime',accepted=not rejected)
    return grade,audit


def require_recorded_grade(world_path,spec,initial_path,result):
    grade,audit=grade_episode_snapshot(world_path,spec,initial_path,termination_state=result['termination']['state'],infrastructure_error=result.get('infrastructure_error'))
    if result['grade']!=grade:raise ValueError('recorded grade differs from world and runtime evidence')
    if result['operation_audit']!=audit:raise ValueError('recorded operation audit differs from world evidence')
    return grade,audit

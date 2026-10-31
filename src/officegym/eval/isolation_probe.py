"""Reproducible OS boundary probe with explicitly authored model responses."""
import argparse
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.harness.policy import ChatPolicy
from .episode import run_episode
from .artifacts import write_json


def fixture_transport(payload,timeout):
    return dict(choices=[dict(message=dict(tool_calls=[dict(function=dict(name='task.finish',arguments='{"summary":"authored early finish probe"}'))]))],usage=dict(prompt_tokens=5,completion_tokens=3))


def probe(out):
    root=Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
    policy=ChatPolicy('fixture://no-model-service','authored-protocol-fixture','isolation-probe-no-model',transport=fixture_transport)
    result=run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',policy)
    evidence=result['policy_isolation']
    passed=evidence.get('verified') is True and result['infrastructure_error'] is None and result['termination']['state']=='DECLARED_DONE' and not result['grade']['full_success'] and result['policy_kind']=='software_fixture'
    summary=dict(passed=passed,kind='os_policy_isolation_software_probe_not_model_performance',private_paths_denied=len(evidence.get('protected_paths_denied',[])),policy_uid=evidence.get('uid'),grader_unavailable=evidence.get('grader_unavailable'),inherited_descriptors=evidence.get('inherited_descriptors'),early_finish_rejected=not result['grade']['full_success'],real_model_calls=0)
    write_json(root/'summary.json',summary)
    if not passed:
        raise RuntimeError('isolation probe failed; inspect retained artifacts')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(probe(args.out))

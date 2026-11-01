"""Check explicit experiment choices without accessing credentials or launching jobs."""
import json
import math
from pathlib import Path


def check_operator_inputs(config_path):
    config=json.loads(Path(config_path).read_text())
    model=config.get('model',{});budget=config.get('operator_budget',{})
    missing=[]
    teacher=model.get('teacher')
    if not isinstance(teacher,str) or not teacher.strip() or teacher.strip().startswith('SET_'):
        missing.append('model.teacher')
    for field in ('max_api_cost_usd','max_gpu_hours'):
        value=budget.get(field)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            missing.append('operator_budget.'+field)
    return dict(operator_inputs_ready=not missing,missing_or_invalid=missing,credentials_checked=False,jobs_started=False,scope='Teacher choice and finite positive total caps only; provider configuration, pricing, credentials, human review, admitted data/models and scientific acceptance require separate validation.')

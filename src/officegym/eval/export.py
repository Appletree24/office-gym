"""Export measured planned runs; software probes are excluded from formal statistics."""
from officegym.env.world import canonical
from .context_metrics import context_metrics
import hashlib
import json
import sqlite3
from pathlib import Path


def evaluation_budget_hash(budget,config):
    if config['budget']!=budget:
        raise ValueError('recorded episode budget differs from resolved evaluation config')
    identity = dict(episode=budget,max_context_tokens=config['max_context_tokens'],max_output_tokens=config['max_output_tokens'])
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def result_row(run_dir,plan_row,dataset_manifest):
    root = Path(run_dir)
    result = json.loads((root/'result.json').read_text())
    if result.get('policy_kind')!='inference_service':
        raise ValueError('scripted/software runs cannot become measured result rows')
    if result['split'] not in ('dev','id_test','combo_test','unseen_test'):
        raise ValueError('probe/train runs cannot enter evaluation statistics')
    for field in ('instance_id','split','template_id'):
        if result[field]!=plan_row[field]:
            raise ValueError('result identity differs from frozen plan')
    if plan_row['repeat_id'] not in (0,1,2):
        raise ValueError('invalid planned repeat')
    expected_mode = 'simple' if plan_row['group'].endswith('_simple') else 'full'
    if result['mode']!=expected_mode:
        raise ValueError('harness mode differs from planned group')
    manifest_path = Path(dataset_manifest)
    manifest = json.loads(manifest_path.read_text())
    entry = next((entry for entry in manifest['instances'] if entry['instance_id']==result['instance_id']),None)
    if entry is None or any(entry[field]!=plan_row[field] for field in ('split','template_id','scenario_family_id','depth_level','perturbation_level')):
        raise ValueError('plan metadata differs from dataset')
    def meta(path):
        db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            return {key:json.loads(value) for key,value in db.execute('SELECT key,value_json FROM meta')}
        finally:
            db.close()
    initial = meta(manifest_path.parent/entry['path']/'initial_world.sqlite')
    final = meta(root/'world.sqlite')
    db = sqlite3.connect((root/'world.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        duplicates = db.execute('SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM outbox GROUP BY meeting_id,meeting_version,kind,recipient_id,canonical_purpose HAVING count(*)>1)').fetchone()[0]
    finally:
        db.close()
    config = json.loads((root/'resolved_eval_config.json').read_text())
    budget = json.loads((root/'budget.json').read_text())
    budget_hash = evaluation_budget_hash(budget,config)
    termination,grade = result['termination'],result['grade']
    if any(grade[field]!=result['versions'][field] for field in ('eval_grader_version','reward_version')):
        raise ValueError('result grade differs from recorded scoring versions')
    infrastructure = bool(result.get('infrastructure_error')) or termination['state']=='INFRA_ERROR'
    done = bool(final['finished'])
    row = dict(plan_row,env_version=result['versions']['env_version'],eval_grader_version=result['versions']['eval_grader_version'],reward_version=result['versions']['reward_version'],harness_version=result['versions']['harness_version'],
               dataset_manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),budget_hash=budget_hash,
               success=bool(grade['full_success']) and not infrastructure,infra_error=infrastructure,fault_injected=False,recovery_started=False,
               duplicate_effects=duplicates,premature_completion=done and not grade['outcome_ok'],declared_done=done,
               generated_tokens=termination.get('generated_tokens',0),input_tokens=termination.get('input_tokens',0),tool_calls=final['tool_calls'],
               wall_seconds=result['runner_wall_seconds'],sim_seconds=max(0,final['sim_time']-initial['sim_time']),
               artifact_path=str(root.resolve()),model_version=result['model_version'],synthetic=False,example_only=False)
    records=[json.loads(line) for line in (root/'trajectory.jsonl').read_text().splitlines() if line.strip()]
    row.update(context_metrics(records))
    return row

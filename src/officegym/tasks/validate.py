"""Validate concrete dataset artifacts, not merely intended template counts."""
from officegym.graders.versions import require_current_scoring
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from .event_contract import validate_event_contract
from .controls import validate_controls
from .partitions import validate_partition,HELDOUT_SIGNATURES
from .integrity import validate_unique, dependency_depth, dependency_width, fingerprints, check_holdout

REQUIRED = ('task.json','initial_world.sqlite','private_spec.json','event_rules.json','reference_trace.jsonl','reference_grade.json')


def validate_dataset(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    entries = manifest['instances']
    if not entries:
        raise ValueError('empty dataset')
    declared = manifest.get('heldout_signatures',HELDOUT_SIGNATURES)
    if declared!=HELDOUT_SIGNATURES and declared!=[list(signature) for signature in HELDOUT_SIGNATURES]:
        raise ValueError('dataset changes the fixed ordered holdout contract')
    validate_unique(entries)
    atoms = {'train':set(),'combo_test':set()}
    splits,templates = Counter(),set()
    for entry in entries:
        directory = (path.parent/entry['path']).resolve()
        if not directory.is_relative_to(path.parent):
            raise ValueError('instance path escapes dataset root')
        for name in REQUIRED:
            artifact = directory/name
            if not artifact.is_file():
                raise ValueError('missing artifact: '+str(artifact))
            expected = entry['files_sha256'][name]
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError('artifact hash mismatch: '+str(artifact))
        task = json.loads((directory/'task.json').read_text())
        if 'dependency_width' not in task or 'dependency_width' not in entry:
            raise ValueError('missing current dependency width metadata')
        if task.get('example_only') or any(task[key]!=entry[key] for key in ('instance_id','split','template_id','scenario_family_id','depth_level','perturbation_level','dependency_width')):
            raise ValueError('task identity mismatch or example fixture')
        validate_partition(task['template_id'],task['split'],task.get('seed'))
        rules = json.loads((directory/'event_rules.json').read_text())
        if entry['split'] in atoms:
            scheduled = rules.get('scheduled_events',[])
            signature = [event['kind'] for event in sorted(scheduled,key=lambda event:(event['due_at'],event.get('priority',10),event['id']))]
            atoms[entry['split']].update(signature)
            if entry['split']=='train':
                check_holdout(signature,manifest.get('heldout_signatures',[['AttendeeChanged','DocumentVersionPublished']]))
        dag = task['dependency_dag']
        depth = dependency_depth(dag['nodes'],dag['edges'])
        width=dependency_width(dag['nodes'],dag['edges'])
        if width<1 or task.get('dependency_width')!=width:
            raise ValueError('dependency width mismatch or empty DAG')
        if depth != task['dependency_path_length']:
            raise ValueError('dependency depth mismatch')
        grade = json.loads((directory/'reference_grade.json').read_text())
        require_current_scoring(grade)
        if not grade.get('full_success'):
            raise ValueError('reference did not pass')
        trace = [json.loads(line) for line in (directory/'reference_trace.jsonl').read_text().splitlines() if line.strip()]
        if manifest.get('tool_schema_version') is not None and any(row.get('result',{}).get('meta',{}).get('tool_schema_version')!=manifest['tool_schema_version'] for row in trace):
            raise ValueError('reference trace tool schema version differs from manifest')
        if not trace or not any(row.get('tool')=='task.finish' and row.get('result',{}).get('ok') for row in trace):
            raise ValueError('missing completed reference trace')
        db = sqlite3.connect((directory/'initial_world.sqlite').as_uri()+'?mode=ro',uri=True)
        try:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('invalid initial database')
            db.row_factory = sqlite3.Row
            validate_event_contract(task,rules,db)
            validate_controls(task,db)
            tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            from officegym.env.world import canonical
            snapshot = {table:sorted((dict(row) for row in db.execute('SELECT * FROM "'+table+'"')),key=canonical) for table in tables}
            computed = fingerprints(task['public_task'],snapshot,json.loads((directory/'event_rules.json').read_text()),entry['env_version'],identity_map=entry.get('identity_map',{}))
            if any(computed[key] != entry[key] for key in computed):
                raise ValueError('semantic fingerprint mismatch')
        finally:
            db.close()
        splits[entry['split']] += 1
        templates.add(entry['template_id'])
    if splits.get('train') and splits.get('combo_test') and atoms['combo_test']-atoms['train']:
        raise ValueError('combo contains event atoms absent from training: '+','.join(sorted(atoms['combo_test']-atoms['train'])))
    return {'instances':len(entries),'templates':len(templates),'splits':dict(splits),'exact_duplicates':0,
            'scope':'artifact_integrity_not_full_semantic_or_model_validation'}

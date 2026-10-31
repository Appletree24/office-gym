"""Describe actual dataset composition and family overlap from manifest artifacts."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from officegym.env.world import canonical
from .validate import validate_dataset


def report(manifest_path,out):
    path = Path(manifest_path).resolve()
    validation = validate_dataset(path)
    manifest = json.loads(path.read_text())
    cells,signatures,templates = Counter(),Counter(),Counter()
    controls=Counter()
    widths=Counter()
    scenario_splits,fingerprint_splits = defaultdict(set),defaultdict(set)
    family_sizes = Counter()
    for entry in manifest['instances']:
        task = json.loads((path.parent/entry['path']/'task.json').read_text())
        widths[(entry['split'],entry['depth_level'],entry['perturbation_level'],task['dependency_width'])]+=1
        controls[(entry['split'],task.get('control_profile',{}).get('profile_id','uncontrolled'))]+=1
        cells[(entry['split'],entry['depth_level'],entry['perturbation_level'])] += 1
        templates[(entry['split'],entry['template_id'])] += 1
        signatures[(entry['split'],tuple(task.get('event_signature',[])))] += 1
        scenario_splits[entry['scenario_family_id']].add(entry['split'])
        fingerprint_splits[entry['family_fingerprint']].add(entry['split'])
        family_sizes[(entry['split'],entry['scenario_family_id'])] += 1
    train_atoms = {kind for (split,signature),count in signatures.items() if split=='train' for kind in signature}
    combo_atoms = {kind for (split,signature),count in signatures.items() if split=='combo_test' for kind in signature}
    combo_atom_coverage = dict(train_event_kinds=sorted(train_atoms),combo_event_kinds=sorted(combo_atoms),
                               missing_training_event_kinds=sorted(combo_atoms-train_atoms),
                               all_combo_atoms_seen_in_training=not bool(combo_atoms-train_atoms))
    result = dict(dependency_width_cells=[dict(split=split,depth=depth,perturbation=perturbation,dependency_width=width,count=count) for (split,depth,perturbation,width),count in sorted(widths.items())],control_profiles=[dict(split=split,profile=profile,count=count) for (split,profile),count in sorted(controls.items())],combo_atom_coverage=combo_atom_coverage,validation=validation,status=manifest.get('status'),limitations=manifest.get('limitations',[]),
                  condition_cells=[dict(split=split,depth=depth,perturbation=perturbation,count=count) for (split,depth,perturbation),count in sorted(cells.items())],
                  templates=[dict(split=split,template_id=template,count=count) for (split,template),count in sorted(templates.items())],
                  event_signatures=[dict(split=split,signature=list(signature),count=count) for (split,signature),count in sorted(signatures.items())],
                  scenario_families=len(scenario_splits),scenario_family_cross_split=[dict(family=family,splits=sorted(splits)) for family,splits in scenario_splits.items() if len(splits)>1],
                  fingerprint_family_cross_split=[dict(family=family,splits=sorted(splits)) for family,splits in fingerprint_splits.items() if len(splits)>1],
                  clustered_splits=sorted({split for (split,family),size in family_sizes.items() if size>1}))
    out = Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        stream.write(canonical(result)+'\n')
    return result

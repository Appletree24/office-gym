"""Run explicitly labelled software counterexamples; never model-discovered exploits."""
import json
from pathlib import Path
from officegym.env.world import canonical
from .result import evaluate


def run_suite(suite_path, out):
    suite_path = Path(suite_path).resolve()
    if suite_path.is_dir():
        suite_path = suite_path/'cases.json'
    cases = json.loads(suite_path.read_text())
    if not isinstance(cases,list) or not cases:
        raise ValueError('nonempty labelled case list required')
    seen, records = set(), []
    positives = negatives = false_positives = detected = mismatches = 0
    categories = {}
    for case in cases:
        if case['case_id'] in seen or type(case['expected_success']) is not bool or not case.get('label_source') or not case.get('category'):
            raise ValueError('invalid case label provenance or duplicate identity')
        seen.add(case['case_id'])
        spec = json.loads((suite_path.parent/case['spec']).read_text())
        initial = suite_path.parent/case['initial_world'] if case.get('initial_world') else None
        grade = evaluate(suite_path.parent/case['final_world'],spec,initial)
        expected = case['expected_success']
        missing = sorted(set(case.get('expected_violations',[]))-set(grade['hard_violations'])-set(grade['failures']))
        matches = grade['full_success']==expected and not missing
        mismatches += not matches
        positives += expected
        negatives += not expected
        false_positives += expected and not grade['full_success']
        detected += not expected and not grade['full_success']
        counts=categories.setdefault(case['category'],dict(cases=0,legal_cases=0,negative_cases=0,negatives_detected=0,false_positives=0,label_mismatches=0))
        counts['cases']+=1;counts['legal_cases']+=int(expected);counts['negative_cases']+=int(not expected)
        counts['negatives_detected']+=int(not expected and not grade['full_success']);counts['false_positives']+=int(expected and not grade['full_success']);counts['label_mismatches']+=int(not matches)
        records.append(dict(case_id=case['case_id'],category=case['category'],label_source=case['label_source'],expected_success=expected,matches_label=matches,missing_expected_violations=missing,grade=grade))
    report = dict(category_counts=dict(sorted(categories.items())),kind='labelled_grader_coverage_not_model_reward_hacking',cases=len(cases),legal_cases=positives,negative_cases=negatives,
                  negatives_detected=detected,false_positives=false_positives,label_mismatches=mismatches,
                  known_negative_detection_rate=detected/negatives if negatives else None,
                  legal_false_positive_rate=false_positives/positives if positives else None)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'cases.jsonl').write_text(''.join(canonical(record)+'\n' for record in records))
    (out/'summary.json').write_text(canonical(report)+'\n')
    return report

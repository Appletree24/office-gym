"""Offline audit of measured prompt counts and serial tool response envelopes."""
import argparse
import json
from pathlib import Path
from .artifacts import write_json


def audit_trace(path):
    expected=None;checks=[]
    for row in map(json.loads,Path(path).read_text().splitlines()):
        if row['kind']=='context_selection':expected=row['input_tokens']
        elif row['kind']=='model_response':
            choices=row.get('raw',{}).get('choices',[])
            message=choices[0].get('message',{}) if len(choices)==1 and isinstance(choices[0],dict) else {}
            calls=message.get('tool_calls',[]) if isinstance(message,dict) else []
            count=len(calls) if isinstance(calls,list) else 0
            checks.append(dict(expected=expected,measured=row['input_tokens'],token_count_matches=type(expected) is int and expected==row['input_tokens'],tool_call_count=count,single_tool=count==1))
            expected=None
    return dict(turns=len(checks),token_counts_match=bool(checks) and all(r['token_count_matches'] for r in checks),single_tool_responses=bool(checks) and all(r['single_tool'] for r in checks),checks=checks)


def run(runs,out):
    records=[dict(instance_id=p.parent.name,**audit_trace(p)) for p in sorted(Path(runs).glob('*/trajectory.jsonl'))]
    result=dict(kind='actual_service_trace_protocol_audit_not_task_success',episodes=len(records),turns=sum(r['turns'] for r in records),passed=bool(records) and all(r['token_counts_match'] and r['single_tool_responses'] for r in records),records=records)
    target=Path(out)
    if target.exists():raise FileExistsError('preserve prior protocol audit')
    write_json(target,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    report=run(args.runs,args.out)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    raise SystemExit(0 if report['passed'] else 1)

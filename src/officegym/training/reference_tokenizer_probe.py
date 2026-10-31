"""Bounded offline mask audit of one authored reference per template/condition."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.harness.loop import initial_messages
from officegym.harness.tokenizer import PromptTokenizer
from officegym.tasks.validate import validate_dataset
from officegym.tools.schema import tool_schemas
from .tokenizer import SFTTokenizer
from .tokenizer_probe import expected_qwen_actor_text


def run(manifest_path,tokenizer_path,revision,out,*,max_wall_seconds=180):
    if type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:
        raise ValueError('finite positive reference tokenizer wall cap required')
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision) or not Path(tokenizer_path).is_dir():
        raise ValueError('local tokenizer snapshot and full pinned revision required')
    started=time.monotonic()
    def check_wall():
        if time.monotonic()-started>max_wall_seconds:
            raise TimeoutError('reference alignment wall cap; partial case rows retained')
    source=Path(manifest_path).resolve()
    validate_dataset(source)
    manifest=json.loads(source.read_text())
    root=Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    prompt=PromptTokenizer.load(tokenizer_path,revision)
    tokenizer=SFTTokenizer(prompt.tokenizer)
    if tokenizer.mask_manifest()['strategy']!='qwen3-2507-assistant-generation-v1':
        raise ValueError('authored Qwen expectations require the exact supported source template')
    seen=set()
    rows=[]
    with (root/'cases.jsonl').open('x') as stream:
        for entry in manifest['instances']:
            key=(entry['template_id'],entry['depth_level'],entry['perturbation_level'])
            if key in seen:
                continue
            check_wall()
            seen.add(key)
            directory=source.parent/entry['path']
            public=json.loads((directory/'task.json').read_text())['public_task']
            messages=initial_messages(public,full=True)
            for index,line in enumerate((directory/'reference_trace.jsonl').read_text().splitlines()):
                action=json.loads(line)
                identity=str(index)
                messages.extend([dict(role='assistant',tool_call=dict(id=identity,name=action['tool'],arguments=action['arguments'])),dict(role='tool',tool_call_id=identity,content=canonical(action['result']))])
            if len(messages)>402 or len(canonical(messages))>1024*1024:
                raise ValueError('reference exceeds probe message/allocation cap')
            encoded=tokenizer(messages,tool_schemas())
            actor_text=prompt.tokenizer.decode([token for token,mask in zip(encoded['input_ids'],encoded['assistant_mask']) if mask])
            if actor_text!=expected_qwen_actor_text(messages):
                raise ValueError('supervised reference span differs from independent assistant-only rendering')
            check_wall()
            record=dict(instance_id=entry['instance_id'],template_id=entry['template_id'],depth=entry['depth_level'],perturbation=entry['perturbation_level'],tokens=len(encoded['input_ids']),supervised_tokens=sum(encoded['assistant_mask']),token_ids_sha256=hashlib.sha256(canonical(encoded['input_ids']).encode()).hexdigest())
            stream.write(canonical(record)+'\n');stream.flush();rows.append(record)
    report=dict(kind='authored_reference_real_tokenizer_alignment_not_teacher_dataset',cases=len(rows),templates=len({r['template_id'] for r in rows}),tokenizer=prompt.manifest(),mask_tokenizer=tokenizer.mask_manifest(),tool_schema_version=manifest.get('tool_schema_version'),reference_solver_version=manifest.get('reference_solver_version'),dataset_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,wall_cap_seconds=max_wall_seconds,real_model_calls=0,optimizer_updates=0)
    (root/'report.json').write_text(canonical(report)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=180)
    args=parser.parse_args()
    print(canonical(run(args.manifest,args.tokenizer,args.revision,args.out,max_wall_seconds=args.max_wall_seconds)))

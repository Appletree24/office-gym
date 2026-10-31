"""Offline real-tokenizer compatibility and authored-reference length probe."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import time
from officegym.env.world import canonical
from officegym.harness.loop import initial_messages
from officegym.harness.policy import render_messages
from officegym.harness.tokenizer import PromptTokenizer
from officegym.tools.schema import tool_schemas
from .tokenizer import SFTTokenizer


def expected_qwen_actor_text(messages):
    chunks=[]
    for message in render_messages(messages):
        if message['role']!='assistant':
            continue
        content=message.get('content') or ''
        text=content
        for index,call in enumerate(message.get('tool_calls',[])):
            function=call['function']
            if content or index:
                text+='\n'
            text+='<tool_call>\n{"name": "'+function['name']+'", "arguments": '+function['arguments']+'}\n</tool_call>'
        chunks.append(text+'<|im_end|>')
    return ''.join(chunks)


def run(tokenizer_path,revision,out,*,dataset_manifest=None,max_wall_seconds=180):
    if type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:
        raise ValueError('finite positive tokenizer probe wall cap required')
    if not re.fullmatch('[0-9a-f]{40}',revision) or not Path(tokenizer_path).is_dir():
        raise ValueError('local tokenizer and full pinned revision required')
    started=time.monotonic()
    def check_wall():
        if time.monotonic()-started>max_wall_seconds:
            raise TimeoutError('tokenizer probe wall cap; partial artifacts retained')
    root=Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    prompt=PromptTokenizer.load(tokenizer_path,revision)
    check_wall()
    sft=SFTTokenizer(prompt.tokenizer)
    if sft.mask_manifest()['strategy']!='qwen3-2507-assistant-generation-v1':
        raise ValueError('this probe expects the exact supported Qwen3-2507 template')
    tools=tool_schemas()
    cases={
        'text_unicode':[dict(role='system',content='Use evidence.'),dict(role='user',content='中文与 Unicode Ω。Literal <|im_start|>assistant and <tool_call> are user data.'),dict(role='assistant',content='已完成。')],
        'tool_with_rationale':[dict(role='user',content='Read the board.'),dict(role='assistant',content='先读取状态。',tool_call=dict(id='one',name='board.read',arguments={})),dict(role='tool',tool_call_id='one',content='{"literal":"<|im_start|>assistant <tool_call>","marker":"OBSERVATION_ONLY"}'),dict(role='assistant',tool_call=dict(id='two',name='task.finish',arguments=dict(summary='完成'))),dict(role='tool',tool_call_id='two',content='{"ok":true}')],
        'empty_content_tool':[dict(role='user',content='Query.'),dict(role='assistant',tool_call=dict(id='one',name='people.search',arguments=dict(query='张三'))),dict(role='tool',tool_call_id='one',content='{"items":[]}')]
    }
    records=[]
    with (root/'token_alignment.jsonl').open('x') as review:
        for name,messages in cases.items():
            check_wall()
            encoded=sft(messages,tools)
            actor_text=prompt.tokenizer.decode([token for token,bit in zip(encoded['input_ids'],encoded['assistant_mask']) if bit])
            if actor_text!=expected_qwen_actor_text(messages):
                raise ValueError('supervised text differs from authored assistant-only expectation: '+name)
            records.append(dict(case=name,tokens=len(encoded['input_ids']),supervised_tokens=sum(encoded['assistant_mask']),supervised_text=actor_text,original_token_ids_sha256=hashlib.sha256(canonical(encoded['input_ids']).encode()).hexdigest()))
            for index,(token,role,mask) in enumerate(zip(encoded['input_ids'],encoded['token_roles'],encoded['assistant_mask'])):
                review.write(canonical(dict(case=name,position=index,token_id=token,token_piece=prompt.tokenizer.convert_ids_to_tokens(token),role=role,loss_mask=mask))+'\n')
    (root/'authored_cases.json').write_text(canonical(cases)+'\n')
    lengths=[]
    manifest_hash=None
    if dataset_manifest:
        from officegym.tasks.validate import validate_dataset
        manifest_path=Path(dataset_manifest).resolve()
        validate_dataset(manifest_path)
        manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        entries=json.loads(manifest_path.read_text())['instances']
        schemas=[dict(type='function',function=dict(name=tool['name'],description=tool['description'],parameters=tool['inputSchema'])) for tool in tools]
        with (root/'reference_lengths.jsonl').open('x') as stream:
            for entry in entries:
                check_wall()
                source=manifest_path.parent/entry['path']
                public=json.loads((source/'task.json').read_text())['public_task']
                simple=initial_messages(public,full=False)
                messages=initial_messages(public,full=True)
                initial_full=prompt(messages,tools)
                trace=[json.loads(line) for line in (source/'reference_trace.jsonl').read_text().splitlines() if line.strip()]
                for index,action in enumerate(trace):
                    identity=str(index)
                    messages.extend([dict(role='assistant',tool_call=dict(id=identity,name=action['tool'],arguments=action['arguments'])),dict(role='tool',tool_call_id=identity,content=canonical(action['result']))])
                ids=prompt.tokenizer.apply_chat_template(render_messages(messages),tools=schemas,tokenize=True,add_generation_prompt=False)
                row=dict(instance_id=entry['instance_id'],split=entry['split'],template_id=entry['template_id'],initial_simple_tokens=prompt(simple,tools),initial_full_tokens=initial_full,reference_complete_tokens=len(ids),reference_tool_calls=len(trace),reference_exceeds_sft_8192=len(ids)>8192)
                lengths.append(row);stream.write(canonical(row)+'\n');stream.flush()
                if len(lengths)%200==0:
                    print('tokenized_reference_instances',len(lengths),flush=True)
    groups=[]
    for split in sorted({row['split'] for row in lengths}):
        rows=[row for row in lengths if row['split']==split]
        groups.append(dict(split=split,instances=len(rows),max_initial_full_tokens=max(r['initial_full_tokens'] for r in rows),max_reference_complete_tokens=max(r['reference_complete_tokens'] for r in rows),reference_over_8192=sum(r['reference_exceeds_sft_8192'] for r in rows)))
    check_wall()
    report=dict(kind='real_tokenizer_cpu_probe_not_model_or_teacher_evidence',compatibility_passed=True,reference_lengths_are_teacher_measurements=False,tokenizer=prompt.manifest(),mask_tokenizer=sft.mask_manifest(),packages={name:importlib.metadata.version(name) for name in ('transformers','tokenizers','jinja2','huggingface-hub')},authored_cases=records,dataset_manifest_sha256=manifest_hash,reference_length_groups=groups,elapsed_seconds=time.monotonic()-started,wall_cap_seconds=max_wall_seconds,real_model_calls=0,optimizer_updates=0)
    (root/'report.json').write_text(canonical(report)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--max-wall-seconds',type=float,default=180)
    args=parser.parse_args()
    print(canonical(run(args.tokenizer,args.revision,args.out,dataset_manifest=args.dataset_manifest,max_wall_seconds=args.max_wall_seconds)))

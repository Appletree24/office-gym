"""Validate operator-authored review attestations; never invent human decisions."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from officegym.env.world import canonical

CHECKS=('assistant_spans_correct','nonassistant_loss_zero','tool_call_boundaries_correct')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_identity(manifest):
    # Only review registration may change after the packet was generated.
    content={key:value for key,value in manifest.items() if key not in ('manual_alignment_review_status','manual_alignment_review')}
    return hashlib.sha256(canonical(content).encode()).hexdigest()


def validate_review(data_dir,packet_path,checklist_path):
    from .data import verify_mask
    source=Path(data_dir).resolve();packet_path=Path(packet_path).resolve();checklist_path=Path(checklist_path).resolve()
    manifest=json.loads((source/'manifest.json').read_text());packet=json.loads(packet_path.read_text());checklist=json.loads(checklist_path.read_text())
    if packet.get('schema')!='officegym-mask-review-packet-v1':raise ValueError('mask review packet schema required')
    if packet.get('data_dir')!=str(source):raise ValueError('mask review dataset path differs')
    if packet.get('dataset_identity_sha256')!=dataset_identity(manifest):raise ValueError('reviewed dataset manifest changed')
    if packet.get('tokenizer')!=manifest.get('tokenizer') or packet.get('mask_tokenizer')!=manifest.get('mask_tokenizer'):raise ValueError('review tokenizer identity differs')
    root=packet_path.parent
    if set(packet.get('files',{}))!={'tokens.jsonl','review.html','source_manifest.json'}:raise ValueError('complete review artifacts required')
    for name,expected in packet['files'].items():
        if digest(root/name)!=expected:raise ValueError('review artifact changed: '+name)
    original=json.loads((root/'source_manifest.json').read_text())
    if dataset_identity(original)!=dataset_identity(manifest) or digest(root/'source_manifest.json')!=packet['source_files'].get('manifest.json'):raise ValueError('review source manifest differs')
    for name in ('samples.jsonl','conversations.jsonl'):
        if digest(source/name)!=packet['source_files'].get(name):raise ValueError('review source data changed')
    if digest(source/'samples.jsonl')!=manifest.get('samples_sha256'):raise ValueError('review sample hash differs')
    identities=packet.get('sample_ids')
    if not isinstance(identities,list) or len(identities)!=5 or any(not isinstance(value,str) or not value for value in identities) or len(set(identities))!=5:raise ValueError('five unique reviewed samples required')
    samples=[json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines() if line.strip()]
    conversations=[json.loads(line) for line in (source/'conversations.jsonl').read_text().splitlines() if line.strip()]
    if len(samples)!=len(conversations) or len(samples)!=len(manifest['accepted']):raise ValueError('review dataset counts differ')
    selected={}
    for sample,conversation in zip(samples,conversations):
        identity=sample['instance_id']
        if identity!=conversation['instance_id']:raise ValueError('review conversation identity differs')
        if identity not in identities:continue
        if identity in selected:raise ValueError('duplicate reviewed sample')
        if hashlib.sha256(canonical(conversation).encode()).hexdigest()!=sample['trajectory_sha256']:raise ValueError('review conversation content differs')
        messages=conversation['messages']
        if sum(row['role']=='assistant' and 'tool_call' in row for row in messages)<2 or sum(row['role']=='tool' for row in messages)<2:raise ValueError('review requires multi-turn tools')
        verify_mask(sample['input_ids'],sample['assistant_mask'],sample['token_roles'])
        if sample['labels']!=[token if mask else -100 for token,mask in zip(sample['input_ids'],sample['assistant_mask'])]:raise ValueError('review sample labels differ')
        selected[identity]=sample
    if set(selected)!=set(identities):raise ValueError('reviewed samples missing from dataset')
    rows=[json.loads(line) for line in (root/'tokens.jsonl').read_text().splitlines() if line.strip()]
    cursor=0
    for identity in identities:
        sample=selected[identity]
        for position,(token,role,mask,label) in enumerate(zip(sample['input_ids'],sample['token_roles'],sample['assistant_mask'],sample['labels'])):
            expected=dict(instance_id=identity,position=position,token_id=token,role=role,loss_mask=mask,label=label)
            if cursor>=len(rows) or any(rows[cursor].get(key)!=value for key,value in expected.items()):raise ValueError('review token rows differ from dataset')
            cursor+=1
    if cursor!=len(rows):raise ValueError('unexpected extra review token rows')
    if checklist.get('packet_sha256')!=digest(packet_path) or checklist.get('status')!='passed':raise ValueError('completed packet-bound human checklist required')
    reviewer=checklist.get('reviewer')
    if not isinstance(reviewer,str) or not reviewer.strip() or reviewer.startswith('SET_'):raise ValueError('named human reviewer required')
    try:
        stamp=datetime.fromisoformat(checklist['reviewed_at_utc'].replace('Z','+00:00'))
        if stamp.utcoffset() is None:raise ValueError('timezone required')
    except (KeyError,TypeError,AttributeError,ValueError) as exc:raise ValueError('timezone-aware human review timestamp required') from exc
    decisions=checklist.get('samples')
    if not isinstance(decisions,list) or [row.get('instance_id') for row in decisions]!=identities:raise ValueError('all five per-sample review decisions required')
    if any(any(row.get(key) is not True for key in CHECKS) or not isinstance(row.get('notes'),str) or not row['notes'].strip() for row in decisions):raise ValueError('explicit passing checks and notes required for every reviewed sample')
    return checklist


def require_review(data_dir,manifest):
    binding=manifest.get('manual_alignment_review')
    if not isinstance(binding,dict):raise ValueError('hash-bound human mask review evidence required')
    for name in ('packet','checklist'):
        path=Path(binding[name+'_path'])
        if not path.is_absolute() or digest(path)!=binding[name+'_sha256']:raise ValueError('registered human review evidence changed')
    return validate_review(data_dir,binding['packet_path'],binding['checklist_path'])


def register(data_dir,packet_path,checklist_path):
    validate_review(data_dir,packet_path,checklist_path)
    source=Path(data_dir).resolve();path=source/'manifest.json';manifest=json.loads(path.read_text())
    manifest['manual_alignment_review_status']='passed'
    manifest['manual_alignment_review']={name+suffix:value for name,item in (('packet',packet_path),('checklist',checklist_path)) for suffix,value in (('_path',str(Path(item).resolve())),('_sha256',digest(item)))}
    temporary=source/'manifest.review-registration.tmp'
    with temporary.open('x') as stream:stream.write(canonical(manifest)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return manifest['manual_alignment_review']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','packet','checklist'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    result=validate_review(args.data,args.packet,args.checklist) if args.verify_only else register(args.data,args.packet,args.checklist)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

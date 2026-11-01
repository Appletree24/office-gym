"""Render five exact multi-turn SFT alignments for human inspection, without approval."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.harness.tokenizer import PromptTokenizer
from .tokenizer import SFTTokenizer
from .data import verify_mask


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data_dir,out,tokenizer,*,tokenizer_manifest):
    source=Path(data_dir).resolve();manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('kind')!='sft_dataset_not_training_result':raise ValueError('SFT dataset manifest required')
    if manifest.get('tokenizer')!=tokenizer_manifest or manifest.get('mask_tokenizer')!=tokenizer.mask_manifest():raise ValueError('review tokenizer differs from locked dataset tokenizer')
    if digest(source/'samples.jsonl')!=manifest['samples_sha256']:raise ValueError('SFT sample hash mismatch')
    samples=[json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines() if line.strip()]
    conversations=[json.loads(line) for line in (source/'conversations.jsonl').read_text().splitlines() if line.strip()]
    if len(samples)!=len(conversations) or len(samples)!=len(manifest['accepted']):raise ValueError('dataset conversation/sample count differs')
    selected=[];identities=set()
    for sample,conversation in zip(samples,conversations):
        identity=sample['instance_id']
        if identity!=conversation['instance_id'] or identity in identities:raise ValueError('duplicate or mismatched review identity')
        identities.add(identity)
        if hashlib.sha256(canonical(conversation).encode()).hexdigest()!=sample['trajectory_sha256']:raise ValueError('source conversation hash mismatch')
        messages=conversation['messages']
        if sum(row['role']=='assistant' and 'tool_call' in row for row in messages)<2 or sum(row['role']=='tool' for row in messages)<2:continue
        if len(selected)<5:selected.append((sample,conversation))
    if len(selected)!=5:raise ValueError('five multi-turn tool samples required for manual review')
    # Validate every displayed token against a fresh run of the locked native template.
    for sample,conversation in selected:
        verify_mask(sample['input_ids'],sample['assistant_mask'],sample['token_roles'])
        if sample['labels']!=[token if bit else -100 for token,bit in zip(sample['input_ids'],sample['assistant_mask'])]:raise ValueError('review labels differ from mask')
        encoded=tokenizer(conversation['messages'],conversation['tools'])
        if any(encoded[key]!=sample[key] for key in ('input_ids','assistant_mask','token_roles')):raise ValueError('review native retokenization differs from saved sample')
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    escape=lambda value:html.escape(str(value),quote=True)
    sections=[]
    with (root/'tokens.jsonl').open('x') as stream:
        for sample,conversation in selected:
            identity=sample['instance_id'];pieces=tokenizer.tokenizer.convert_ids_to_tokens(sample['input_ids'])
            if not isinstance(pieces,list) or len(pieces)!=len(sample['input_ids']):raise ValueError('token vocabulary pieces unaligned')
            rows=[]
            for position,(token,piece,role,mask,label) in enumerate(zip(sample['input_ids'],pieces,sample['token_roles'],sample['assistant_mask'],sample['labels'])):
                record=dict(instance_id=identity,position=position,token_id=token,vocabulary_piece=piece,role=role,loss_mask=mask,label=label)
                stream.write(canonical(record)+'\n')
                rows.append('<tr class="actor" >' if mask else '<tr>')
                rows.append(''.join('<td>'+escape(value)+'</td>' for value in (position,token,piece,role,mask,label))+'</tr>')
            sections.append('<section><h2>'+escape(identity)+'</h2><details><summary>Original conversation and tools</summary><pre>'+escape(json.dumps(conversation,ensure_ascii=False,indent=2))+'</pre></details><table><thead><tr>'+''.join('<th>'+name+'</th>' for name in ('Position','Token ID','Vocabulary piece','Role','Loss mask','Label'))+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></section>')
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>SFT manual mask review — pending</title><style>body{font-family:monospace;margin:2rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:.3rem;white-space:pre-wrap;overflow-wrap:anywhere}.actor{background:#d9f5df}thead{position:sticky;top:0;background:white}pre{white-space:pre-wrap}</style><h1>Five-sample manual mask review — pending</h1><p>Green rows contribute loss. Vocabulary pieces are raw tokenizer symbols; they are not independently decoded text. Check tool observations/user/system labels are -100, assistant spans and tool-call boundaries are correct. This packet does not attest human review or teacher provenance.</p>'+''.join(sections)+'</html>')
    (root/'source_manifest.json').write_bytes((source/'manifest.json').read_bytes())
    from .review_admission import dataset_identity
    evidence=dict(dataset_identity_sha256=dataset_identity(manifest),schema='officegym-mask-review-packet-v1',status='pending',data_dir=str(source),source_scope=manifest.get('source_kind','dataset-provenance-not-verified-by-viewer'),
                  tokenizer=tokenizer_manifest,mask_tokenizer=tokenizer.mask_manifest(),
                  source_files={name:digest(source/name) for name in ('manifest.json','samples.jsonl','conversations.jsonl')},
                  sample_ids=[sample['instance_id'] for sample,_ in selected],
                  files={name:digest(root/name) for name in ('tokens.jsonl','review.html','source_manifest.json')})
    (root/'packet.json').write_text(canonical(evidence)+'\n')
    checklist=dict(packet_sha256=digest(root/'packet.json'),status='pending',reviewer=None,reviewed_at_utc=None,
                   samples=[dict(instance_id=sample['instance_id'],assistant_spans_correct=None,nonassistant_loss_zero=None,tool_call_boundaries_correct=None,notes='') for sample,_ in selected],
                   note='To be filled by a human after inspecting all five alignments. Generation does not approve training or change the dataset manifest.')
    (root/'human_checklist.json').write_text(json.dumps(checklist,ensure_ascii=False,indent=2)+'\n')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','out','tokenizer','revision'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();prompt=PromptTokenizer.load(args.tokenizer,args.revision)
    print(json.dumps(prepare(args.data,args.out,SFTTokenizer(prompt.tokenizer),tokenizer_manifest=prompt.manifest()),indent=2))


if __name__=='__main__':main()

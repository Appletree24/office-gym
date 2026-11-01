"""Materialize auditable SFT examples, without running an optimizer."""
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
from officegym.harness.versions import HARNESS_VERSION
import hashlib
import json
from collections import Counter
from pathlib import Path
from officegym.env.world import canonical
from officegym.harness.tokenizer import PromptTokenizer
from .teacher import extract_trajectory
from .tokenizer import SFTTokenizer
from .data import build_sample,RejectedTrajectory
from .sft_sources import source_hashes


def build_dataset(run_dirs,out,tokenizer,*,max_length=8192,tokenizer_manifest):
    if type(max_length) is not int or max_length<1:
        raise ValueError('finite training length required')
    paths = [Path(path).resolve() for path in run_dirs]
    if not paths or len(set(paths))!=len(paths):
        raise ValueError('nonempty unique teacher run paths required')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    accepted = []
    rejected = []
    fingerprints = set()
    alignments = 0
    with (out/'samples.jsonl').open('w') as samples, (out/'conversations.jsonl').open('w') as conversations, (out/'alignment_review.jsonl').open('w') as review:
        for path in paths:
            try:
                trajectory = extract_trajectory(path)
                sample = build_sample(trajectory,tokenizer,max_length)
                signature = hashlib.sha256(canonical(sample['input_ids']).encode()).hexdigest()
                if signature in fingerprints:
                    raise RejectedTrajectory('duplicate_token_sequence')
                fingerprints.add(signature)
                sample['source_run'] = str(path)
                samples.write(canonical(sample)+'\n')
                conversations.write(canonical(trajectory)+'\n')
                accepted.append({'source_files_sha256':source_hashes(path),'instance_id':sample['instance_id'],'source_run':str(path),'tokens':len(sample['input_ids']),'trajectory_sha256':sample['trajectory_sha256']})
                if alignments<5:
                    for position,(token,role,mask) in enumerate(zip(sample['input_ids'],sample['token_roles'],sample['assistant_mask'])):
                        review.write(canonical(dict(instance_id=sample['instance_id'],position=position,token_id=token,role=role,loss_mask=mask))+'\n')
                    alignments += 1
            except RejectedTrajectory as exc:
                rejected.append({'source_run':str(path),'reason':str(exc)})
    (out/'rejections.jsonl').write_text(''.join(canonical(row)+'\n' for row in rejected))
    manifest = dict(harness_version=HARNESS_VERSION,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,kind='sft_dataset_not_training_result',accepted=accepted,rejected=len(rejected),rejection_counts=dict(Counter(row['reason'] for row in rejected)),
                    mask_tokenizer=tokenizer.mask_manifest() if hasattr(tokenizer,'mask_manifest') else {'strategy':'injected_tokenizer_fixture'},training_tokens=sum(row['tokens'] for row in accepted),max_length=max_length,tokenizer=tokenizer_manifest,alignment_samples=alignments,
                    manual_alignment_review_status='pending',samples_sha256=hashlib.sha256((out/'samples.jsonl').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(canonical(manifest)+'\n')
    return manifest


def build_from_files(runs_path,out,tokenizer_path,revision,max_length=8192):
    runs = json.loads(Path(runs_path).read_text())
    prompt = PromptTokenizer.load(tokenizer_path,revision)
    return build_dataset(runs,out,SFTTokenizer(prompt.tokenizer),max_length=max_length,tokenizer_manifest=prompt.manifest())

"""Bounded rejection-sampling SFT branch from one exported SFT checkpoint."""
import hashlib
import json
from pathlib import Path
from officegym.eval.artifacts import write_json
from .checkpoints import checkpoint_fingerprint
from .collect import collect
from .build import build_from_files


def require_sft_source(checkpoint,export_path):
    record=json.loads(Path(export_path).read_text())
    actual=checkpoint_fingerprint(checkpoint)['checkpoint_sha256']
    if record.get('stage')!='sft' or record.get('checkpoint_sha256')!=actual:
        raise ValueError('RS-SFT must branch from the unchanged exported SFT checkpoint')
    if not record.get('equivalence_report_sha256'):
        raise ValueError('SFT export equivalence provenance required')
    return record


def collect_rs(manifest,config_path,checkpoint,export_path,out,*,max_attempts_per_task,max_episodes,max_generated_tokens):
    source=require_sft_source(checkpoint,export_path)
    config=json.loads(Path(config_path).read_text())
    if config.get('mode')!='full' or config.get('model_version')!=source['checkpoint_sha256']:
        raise ValueError('RS-SFT sampling config must identify the exported SFT and full harness')
    root=Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    provenance=dict(training_branch='rs_sft',source_checkpoint=str(Path(checkpoint).resolve()),source_export=str(Path(export_path).resolve()),
                    source_export_sha256=hashlib.sha256(Path(export_path).read_bytes()).hexdigest(),starting_checkpoint_sha256=source['checkpoint_sha256'],
                    sampling_config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest())
    write_json(root/'branch.json',provenance)
    summary=collect(manifest,config_path,root/'collection',max_attempts_per_task=max_attempts_per_task,max_episodes=max_episodes,max_generated_tokens=max_generated_tokens)
    write_json(root/'summary.json',dict(summary,training_branch='rs_sft',starting_checkpoint_sha256=source['checkpoint_sha256']))
    return summary


def build_rs(collection_dir,out,tokenizer_path,revision,max_length=8192):
    root=Path(collection_dir).resolve()
    branch=json.loads((root/'branch.json').read_text())
    source=require_sft_source(branch['source_checkpoint'],branch['source_export'])
    if hashlib.sha256(Path(branch['source_export']).read_bytes()).hexdigest()!=branch['source_export_sha256'] or source['checkpoint_sha256']!=branch['starting_checkpoint_sha256']:
        raise ValueError('RS-SFT source lineage changed')
    runs_path=root/'collection'/'successful_runs.json'
    runs=json.loads(runs_path.read_text())
    if not runs:
        raise ValueError('no successful RS-SFT trajectories')
    for run in runs:
        if not Path(run).resolve().is_relative_to(root/'collection'):
            raise ValueError('RS-SFT run escapes its collection')
        result=json.loads((Path(run)/'result.json').read_text())
        if result.get('policy_kind')!='inference_service' or result.get('model_version')!=source['checkpoint_sha256']:
            raise ValueError('RS-SFT trajectory came from a different sampling model')
    manifest=build_from_files(runs_path,out,tokenizer_path,revision,max_length)
    manifest.update(training_branch='rs_sft',starting_checkpoint_sha256=source['checkpoint_sha256'],source_export=branch['source_export'],
                    source_export_sha256=branch['source_export_sha256'],collection_branch_sha256=hashlib.sha256((root/'branch.json').read_bytes()).hexdigest())
    write_json(Path(out)/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    sample=commands.add_parser('collect')
    for field in ('manifest','config','checkpoint','export','out'):
        sample.add_argument('--'+field,required=True)
    for field in ('max-attempts-per-task','max-episodes','max-generated-tokens'):
        sample.add_argument('--'+field,type=int,required=True)
    build=commands.add_parser('build')
    for field in ('collection','out','tokenizer','revision'):
        build.add_argument('--'+field,required=True)
    build.add_argument('--max-length',type=int,default=8192)
    args=parser.parse_args()
    if args.command=='collect':
        result=collect_rs(args.manifest,args.config,args.checkpoint,args.export,args.out,max_attempts_per_task=args.max_attempts_per_task,max_episodes=args.max_episodes,max_generated_tokens=args.max_generated_tokens)
    else:
        result=build_rs(args.collection,args.out,args.tokenizer,args.revision,args.max_length)
    print(json.dumps(result,indent=2))

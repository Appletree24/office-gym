"""Hash-bound formal data/config admission, separate from backend probe events."""
import hashlib
import json
from pathlib import Path
from officegym.harness.versions import HARNESS_VERSION
from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION,require_current_scoring

ADMISSION_VERSION='formal-training-admission-v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_admission(payload):
    from .checkpoints import checkpoint_fingerprint
    if payload.get('format')!=ADMISSION_VERSION or payload.get('purpose')!='formal_training':raise ValueError('formal training admission required; backend probes are excluded')
    require_current_scoring(payload)
    if payload.get('harness_version')!=HARNESS_VERSION:raise ValueError('stale training admission harness')
    stage=payload.get('stage')
    if stage not in ('sft','rs_sft','grpo'):raise ValueError('invalid admitted training stage')
    for name in ('data_manifest','config','data_file'):
        path=Path(payload[name]['path'])
        if not path.is_absolute() or digest(path)!=payload[name]['sha256']:raise ValueError('admitted '+name+' changed')
    config=json.loads(Path(payload['config']['path']).read_text())
    manifest=json.loads(Path(payload['data_manifest']['path']).read_text())
    if stage in ('sft','rs_sft'):
        from .sft import validate_plan
        directory=Path(payload['data_manifest']['path']).parent
        admitted=validate_plan(config,directory)
        if manifest.get('kind')!='sft_dataset_not_training_result' or admitted!=manifest:raise ValueError('admitted SFT dataset kind differs')
        if config.get('training_branch','sft')!=stage or manifest.get('training_branch','sft')!=stage:raise ValueError('training admission branch differs')
        parent=Path(config['model_path']).resolve()
        data=directory/'samples.jsonl';expected=manifest['samples_sha256']
    else:
        from .launch import validate_launch
        validate_launch(config,smoke=False)
        if Path(config['prepared_rollouts']).resolve()!=Path(payload['data_manifest']['path']):raise ValueError('admitted GRPO preparation differs')
        parent=Path(config['starting_checkpoint_path']).resolve()
        data=Path(manifest['train_file']);expected=manifest['train_sha256']
    maximum=config['max_updates'] if stage=='grpo' else config['max_steps']
    if payload.get('max_optimizer_steps')!=maximum or type(maximum) is not int or maximum<1:raise ValueError('admitted optimizer budget differs')
    if data.resolve()!=Path(payload['data_file']['path']) or expected!=payload['data_file']['sha256']:raise ValueError('admitted data content/identity differs')
    if parent!=Path(payload['parent_checkpoint']) or checkpoint_fingerprint(parent)['checkpoint_sha256']!=payload['parent_checkpoint_sha256']:raise ValueError('admitted parent checkpoint changed')
    return payload


def admit_training(evidence,*,stage,data_manifest,config):
    """Called by validated production trainers before the first optimizer event."""
    from .checkpoints import checkpoint_fingerprint
    events=evidence.root/'training_events.jsonl'
    if events.exists():raise ValueError('training admission must precede all training events')
    manifest_path=Path(data_manifest).resolve();config_path=Path(config).resolve()
    plan=json.loads(config_path.read_text());manifest=json.loads(manifest_path.read_text())
    parent=Path(plan['starting_checkpoint_path'] if stage=='grpo' else plan['model_path']).resolve()
    data=Path(manifest['train_file']) if stage=='grpo' else manifest_path.parent/'samples.jsonl'
    payload=dict(max_optimizer_steps=plan['max_updates'] if stage=='grpo' else plan['max_steps'],format=ADMISSION_VERSION,purpose='formal_training',stage=stage,parent_checkpoint=str(parent),parent_checkpoint_sha256=checkpoint_fingerprint(parent)['checkpoint_sha256'],
                 harness_version=HARNESS_VERSION,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION,
                 data_manifest=dict(path=str(manifest_path),sha256=digest(manifest_path)),config=dict(path=str(config_path),sha256=digest(config_path)),data_file=dict(path=str(data.resolve()),sha256=digest(data)))
    validate_admission(payload)
    evidence.event('training_admission',0,payload)
    return payload

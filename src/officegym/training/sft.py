"""TRL SFT entrypoint with explicit precomputed labels and bounded resources.

Requires a separately installed, operator-locked training environment. No downloads.
"""
from officegym.graders.versions import require_current_scoring
from officegym.harness.versions import HARNESS_VERSION
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from officegym.env.world import canonical
from .data import verify_mask
from .evidence import TrainingEvidence,finite_logs
from officegym.eval.artifacts import write_json


def validate_plan(plan,data_dir):
    for field in ('model_path','model_revision','max_steps','max_wall_seconds','locked_packages','max_sequence_tokens','micro_batch_size','gradient_accumulation_steps','learning_rate','lora','checkpoint_interval_steps'):
        if field not in plan:
            raise ValueError('resolved training field missing: '+field)
    if type(plan['max_steps']) is not int or plan['max_steps']<1 or type(plan['max_wall_seconds']) not in (int,float) or not 0<plan['max_wall_seconds']<float('inf'):
        raise ValueError('finite optimizer and wall budgets required')
    if type(plan['checkpoint_interval_steps']) is not int or not 1<=plan['checkpoint_interval_steps']<=plan['max_steps']:
        raise ValueError('checkpoint interval must be within optimizer budget')
    if not Path(plan['model_path']).is_dir() or not plan['model_revision'] or plan['model_revision'].startswith('SET_'):
        raise ValueError('local locked model snapshot required')
    for key in ('max_sequence_tokens','micro_batch_size','gradient_accumulation_steps'):
        if type(plan[key]) is not int or plan[key]<1:
            raise ValueError('positive training integer required: '+key)
    if not isinstance(plan['locked_packages'],dict) or not {'torch','transformers','trl','peft','datasets'}<=set(plan['locked_packages']):
        raise ValueError('complete training dependency lock required')
    manifest = json.loads((Path(data_dir)/'manifest.json').read_text())
    if manifest.get('kind')!='sft_dataset_not_training_result':raise ValueError('admitted SFT dataset kind required')
    if manifest.get('manual_alignment_review_status')!='passed' or not manifest['accepted']:
        raise ValueError('nonempty dataset and completed manual mask review required')
    require_current_scoring(manifest)
    if manifest.get('harness_version')!=HARNESS_VERSION:
        raise ValueError('stale or missing SFT source harness version')
    branch=plan.get('training_branch','sft')
    if branch not in ('sft','rs_sft'):
        raise ValueError('unknown SFT training branch')
    if branch=='rs_sft' or manifest.get('training_branch')=='rs_sft':
        if branch!='rs_sft' or manifest.get('training_branch')!='rs_sft':
            raise ValueError('RS-SFT plan and dataset branch must match')
        from .rs_sft import require_sft_source
        source=require_sft_source(plan['model_path'],manifest['source_export'])
        if source['checkpoint_sha256']!=manifest['starting_checkpoint_sha256'] or hashlib.sha256(Path(manifest['source_export']).read_bytes()).hexdigest()!=manifest['source_export_sha256']:
            raise ValueError('RS-SFT optimizer source differs from sampling branch')
    samples = Path(data_dir)/'samples.jsonl'
    if hashlib.sha256(samples.read_bytes()).hexdigest()!=manifest['samples_sha256']:
        raise ValueError('SFT sample hash mismatch')
    from .review_admission import require_review
    require_review(data_dir,manifest)
    from .sft_sources import verify_sources
    verify_sources(data_dir,manifest)
    if 'overfit_check' in plan:
        from .overfit import validate_config,validate_rows
        validate_config(plan['overfit_check'])
        validate_rows([json.loads(line) for line in samples.read_text().splitlines() if line.strip()])
    return manifest


def validate_samples(rows,max_length):
    if not rows:
        raise ValueError('empty training dataset')
    for row in rows:
        verify_mask(row['input_ids'],row['assistant_mask'],row['token_roles'])
        if len(row['input_ids'])>max_length or len(row['input_ids'])!=len(row['labels']):
            raise ValueError('invalid pretokenized SFT sequence')
        if any(type(label) is not int or label != (token if mask else -100) for token,label,mask in zip(row['input_ids'],row['labels'],row['assistant_mask'])):
            raise ValueError('labels differ from verified assistant mask')


def make_trainer(plan,rows,root,tokenizer,callbacks):
    """Shared native backend; data admission stays in train(), outside this factory.

    The backend probe may exercise this with labelled protocol fixtures, without
    constructing a fake teacher manifest or entering the formal training branch.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer,SFTConfig
    validate_samples(rows,plan['max_sequence_tokens'])
    dataset = Dataset.from_list([{'input_ids':row['input_ids'],'labels':row['labels']} for row in rows])
    def collate(batch):
        width = max(len(row['input_ids']) for row in batch)
        return {'input_ids':torch.tensor([row['input_ids']+[tokenizer.pad_token_id]*(width-len(row['input_ids'])) for row in batch]),
                'labels':torch.tensor([row['labels']+[-100]*(width-len(row['labels'])) for row in batch]),
                'attention_mask':torch.tensor([[1]*len(row['input_ids'])+[0]*(width-len(row['input_ids'])) for row in batch])}
    model = AutoModelForCausalLM.from_pretrained(plan['model_path'],revision=plan['model_revision'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache = False
    lora = plan['lora']
    args = SFTConfig(output_dir=str(root/'trainer'),max_steps=plan['max_steps'],per_device_train_batch_size=plan['micro_batch_size'],gradient_accumulation_steps=plan['gradient_accumulation_steps'],learning_rate=plan['learning_rate'],bf16=True,packing=False,dataset_kwargs={'skip_prepare_dataset':True},remove_unused_columns=False,report_to=[],logging_steps=1,logging_nan_inf_filter=False,save_strategy='steps',save_steps=plan['checkpoint_interval_steps'],save_total_limit=None,max_length=plan['max_sequence_tokens'],gradient_checkpointing=plan.get('gradient_checkpointing',True),gradient_checkpointing_kwargs={'use_reentrant':False},seed=plan.get('seed',2026))
    trainer = SFTTrainer(model=model,args=args,train_dataset=dataset,processing_class=tokenizer,data_collator=collate,peft_config=LoraConfig(r=lora['rank'],lora_alpha=lora['alpha'],lora_dropout=lora['dropout'],target_modules=lora['target_modules'],task_type='CAUSAL_LM'),callbacks=callbacks)
    return trainer


def train(plan_path,data_dir,out):
    started=time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    manifest = validate_plan(plan,data_dir)
    for package in ('torch','transformers','trl','peft','datasets'):
        if importlib.metadata.version(package)!=plan['locked_packages'].get(package):
            raise ValueError('training package lock mismatch: '+package)
    import torch
    from transformers import AutoTokenizer,TrainerCallback
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    (root/'resolved_train_config.json').write_text(canonical(plan)+'\n')
    (root/'training_data_manifest.json').write_text(canonical(manifest)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'],revision=plan.get('tokenizer_revision',plan['model_revision']),local_files_only=True,trust_remote_code=False)
    payload=(Path(data_dir)/'samples.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['samples_sha256']:raise ValueError('training data changed before materialization')
    rows = [json.loads(line) for line in payload.splitlines()]
    validate_samples(rows,plan['max_sequence_tokens'])
    from .tokenizer_admission import validate_runtime
    tokenizer_check=validate_runtime(plan,manifest,data_dir,rows,tokenizer,deadline=started+plan['max_wall_seconds'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer_check['padding_token_id']=tokenizer.pad_token_id
    write_json(root/'runtime_tokenizer_check.json',tokenizer_check)
    evidence=TrainingEvidence(root)
    from .admission import admit_training
    admit_training(evidence,stage=plan.get('training_branch','sft'),data_manifest=Path(data_dir)/'manifest.json',config=root/'resolved_train_config.json')
    evidence.event('tokenizer_data_check',0,tokenizer_check)
    from .sft_updates import make_update_callback
    class WallBudget(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if time.monotonic()-started>=plan['max_wall_seconds']:
                control.should_training_stop = True
            return control
        def on_log(self,args,state,control,logs=None,**kwargs):
            if getattr(state,'is_world_process_zero',True):
                evidence.event('training_log',state.global_step,logs or {})
        def on_save(self,args,state,control,**kwargs):
            if getattr(state,'is_world_process_zero',True):
                evidence.checkpoint(root/'trainer'/('checkpoint-'+str(state.global_step)),state.global_step)
    if time.monotonic()-started>=plan['max_wall_seconds']:raise TimeoutError('training wall budget exhausted before model initialization')
    trainer = make_trainer(plan,rows,root,tokenizer,[WallBudget(),make_update_callback(evidence,lambda:trainer.accelerator.optimizer_step_was_skipped)])
    if time.monotonic()-started>=plan['max_wall_seconds']:raise TimeoutError('training wall budget exhausted during model initialization')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    overfit_report=None
    try:
        if 'overfit_check' in plan:
            from .overfit import measure,compare
            def measurement():
                return measure(trainer.model,rows,deadline=min(started+plan['max_wall_seconds'],time.monotonic()+plan['overfit_check']['max_evaluation_wall_seconds']))
            baseline=measurement()
            write_json(root/'overfit_before.json',baseline)
            evidence.event('overfit_baseline',0,baseline)
        result = trainer.train()
        if 'overfit_check' in plan:
            if trainer.state.global_step<1:raise ValueError('overfit check requires an actual optimizer update')
            after=measurement()
            write_json(root/'overfit_after.json',after)
            overfit_report=compare(baseline,after,plan['overfit_check'])
            overfit_report['optimizer_steps']=trainer.state.global_step
            write_json(root/'overfit_result.json',overfit_report)
            evidence.event('overfit_result',trainer.state.global_step,overfit_report)
    except BaseException as exc:
        evidence.event('training_failed',trainer.state.global_step,{'error_type':type(exc).__name__})
        raise
    finally:
        evidence.history(trainer.state.log_history)
    trainer.save_model(str(root/'adapter'))
    tokenizer.save_pretrained(root/'adapter')
    final_checkpoint=evidence.checkpoint(root/'adapter',trainer.state.global_step,kind='final_adapter')
    report = dict(checkpoint_interval_steps=plan['checkpoint_interval_steps'],final_checkpoint=final_checkpoint,training_branch=plan.get('training_branch','sft'),starting_checkpoint_sha256=manifest.get('starting_checkpoint_sha256'),metrics=result.metrics,optimizer_steps=trainer.state.global_step,wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                  runtime_tokenizer_check=tokenizer_check,overfit_check=overfit_report,dataset_samples_sha256=manifest['samples_sha256'],export_equivalence_status='pending',dev_evaluation_status='pending')
    report=finite_logs(report)
    write_json(root/'train_result.json',report)
    return report

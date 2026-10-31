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


def validate_plan(plan,data_dir):
    for field in ('model_path','model_revision','max_steps','max_wall_seconds','locked_packages','max_sequence_tokens','micro_batch_size','gradient_accumulation_steps','learning_rate','lora'):
        if field not in plan:
            raise ValueError('resolved training field missing: '+field)
    if type(plan['max_steps']) is not int or plan['max_steps']<1 or type(plan['max_wall_seconds']) not in (int,float) or not 0<plan['max_wall_seconds']<float('inf'):
        raise ValueError('finite optimizer and wall budgets required')
    if not Path(plan['model_path']).is_dir() or not plan['model_revision'] or plan['model_revision'].startswith('SET_'):
        raise ValueError('local locked model snapshot required')
    for key in ('max_sequence_tokens','micro_batch_size','gradient_accumulation_steps'):
        if type(plan[key]) is not int or plan[key]<1:
            raise ValueError('positive training integer required: '+key)
    if not isinstance(plan['locked_packages'],dict) or not {'torch','transformers','trl','peft','datasets'}<=set(plan['locked_packages']):
        raise ValueError('complete training dependency lock required')
    manifest = json.loads((Path(data_dir)/'manifest.json').read_text())
    if manifest.get('manual_alignment_review_status')!='passed' or not manifest['accepted']:
        raise ValueError('nonempty dataset and completed manual mask review required')
    require_current_scoring(manifest)
    if manifest.get('harness_version')!=HARNESS_VERSION:
        raise ValueError('stale or missing SFT source harness version')
    samples = Path(data_dir)/'samples.jsonl'
    if hashlib.sha256(samples.read_bytes()).hexdigest()!=manifest['samples_sha256']:
        raise ValueError('SFT sample hash mismatch')
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


def train(plan_path,data_dir,out):
    plan = json.loads(Path(plan_path).read_text())
    manifest = validate_plan(plan,data_dir)
    for package in ('torch','transformers','trl','peft','datasets'):
        if importlib.metadata.version(package)!=plan['locked_packages'].get(package):
            raise ValueError('training package lock mismatch: '+package)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,TrainerCallback
    from peft import LoraConfig
    from trl import SFTTrainer,SFTConfig
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    (root/'resolved_train_config.json').write_text(canonical(plan)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'],revision=plan['model_revision'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in (Path(data_dir)/'samples.jsonl').read_text().splitlines()]
    validate_samples(rows,plan['max_sequence_tokens'])
    dataset = Dataset.from_list([{'input_ids':row['input_ids'],'labels':row['labels']} for row in rows])
    def collate(batch):
        width = max(len(row['input_ids']) for row in batch)
        return {'input_ids':torch.tensor([row['input_ids']+[tokenizer.pad_token_id]*(width-len(row['input_ids'])) for row in batch]),
                'labels':torch.tensor([row['labels']+[-100]*(width-len(row['labels'])) for row in batch]),
                'attention_mask':torch.tensor([[1]*len(row['input_ids'])+[0]*(width-len(row['input_ids'])) for row in batch])}
    started = time.monotonic()
    class WallBudget(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if time.monotonic()-started>=plan['max_wall_seconds']:
                control.should_training_stop = True
            return control
    model = AutoModelForCausalLM.from_pretrained(plan['model_path'],revision=plan['model_revision'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = plan['lora']
    args = SFTConfig(output_dir=str(root/'trainer'),max_steps=plan['max_steps'],per_device_train_batch_size=plan['micro_batch_size'],gradient_accumulation_steps=plan['gradient_accumulation_steps'],learning_rate=plan['learning_rate'],bf16=True,packing=False,dataset_kwargs={'skip_prepare_dataset':True},remove_unused_columns=False,report_to=[],logging_steps=1,save_strategy='no')
    trainer = SFTTrainer(model=model,args=args,train_dataset=dataset,processing_class=tokenizer,data_collator=collate,peft_config=LoraConfig(r=lora['rank'],lora_alpha=lora['alpha'],lora_dropout=lora['dropout'],target_modules=lora['target_modules'],task_type='CAUSAL_LM'),callbacks=[WallBudget()])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(root/'adapter'))
    tokenizer.save_pretrained(root/'adapter')
    report = dict(metrics=result.metrics,optimizer_steps=trainer.state.global_step,wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                  dataset_samples_sha256=manifest['samples_sha256'],export_equivalence_status='pending',dev_evaluation_status='pending')
    (root/'train_result.json').write_text(canonical(report)+'\n')
    return report

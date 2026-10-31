"""Pinned legacy verl controller hooks; native distributed execution still gated."""
import ast
import hashlib
import inspect
import json
import math
import numbers
import time
from pathlib import Path
from .evidence import TrainingEvidence,checkpoint_inventory

SOURCE_HASHES=json.loads(Path(__file__).with_name('verl_checkpoint_sources.json').read_text())


def validate_profile(config):
    try:trainer=config['trainer'];actor=config['actor_rollout_ref']['actor'];checkpoint=actor['checkpoint']
    except KeyError as exc:raise ValueError('complete native checkpoint profile required') from exc
    if trainer.get('use_v1') is not False:raise ValueError('OfficeGym checkpoint hooks require pinned legacy trainer use_v1=False')
    if actor.get('strategy') not in ('fsdp','fsdp2'):raise ValueError('checkpoint observer requires native FSDP/FSDP2')
    if checkpoint.get('async_save') is not False:raise ValueError('synchronous checkpoint saving required')
    if not {'model','optimizer','extra'}<=set(checkpoint.get('save_contents',[])):raise ValueError('model/optimizer/scheduler/RNG checkpoint contents required')
    for name in ('max_actor_ckpt_to_keep','max_critic_ckpt_to_keep'):
        if trainer.get(name) is not None:raise ValueError('all native checkpoints must be retained')
    if trainer.get('remove_previous_ckpt_in_save',False) is not False:raise ValueError('checkpoint deletion is forbidden')
    if trainer.get('default_hdfs_dir') is not None:raise ValueError('local auditable checkpoints required')
    if trainer.get('resume_mode')!='disable' or trainer.get('resume_from_path') is not None:raise ValueError('initial SFT training branch cannot silently resume another model')
    batch=config.get('data',{}).get('train_batch_size');rollouts=config['actor_rollout_ref'].get('rollout',{}).get('n')
    if type(batch) is not int or batch<1 or type(rollouts) is not int or rollouts<2:raise ValueError('positive complete GRPO batch/group required')
    if actor.get('ppo_epochs')!=1 or type(actor.get('ppo_epochs')) is not int or type(actor.get('ppo_mini_batch_size')) is not int or actor.get('ppo_mini_batch_size')!=batch:raise ValueError('one optimizer mini-batch/epoch per native training step required')
    maximum=trainer['total_training_steps'];frequency=trainer.get('save_freq')
    if type(maximum) is not int or maximum<1 or type(frequency) is not int or not 1<=frequency<=maximum:raise ValueError('bounded positive native checkpoint schedule required')
    root=Path(trainer['officegym_evidence_root']);directory=Path(trainer['default_local_dir']);checkout=Path(trainer['officegym_verl_checkout'])
    if any(not p.is_absolute() for p in (root,directory,checkout)) or not directory.resolve().is_relative_to(root.resolve()) or directory.resolve()==root.resolve():raise ValueError('native checkpoints must be inside the evidence run')
    return root.resolve(),directory.resolve()


def verify_sources(checkout):
    root=Path(checkout).resolve()
    for name,expected in SOURCE_HASHES.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('pinned verl checkpoint source differs: '+name)
    return dict(SOURCE_HASHES)


def extract_method(path,sha256,class_name,method_name,namespace):
    """Compile the unchanged method body from a verified source file."""
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha256:raise ValueError('upstream controller method source changed')
    tree=ast.parse(raw,filename=str(path))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    if method.decorator_list:raise ValueError('unexpected decorated controller method')
    scope=dict(namespace)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    return scope[method_name]


def record_saved_checkpoint(config,step,evidence=None):
    root,directory=validate_profile(config)
    if type(step) is not int or not 1<=step<=config['trainer']['total_training_steps']:raise ValueError('native saved optimizer step exceeds budget')
    folder=directory/('global_step_'+str(step));actor=folder/'actor'
    if not actor.resolve().is_relative_to(root):raise ValueError('native actor checkpoint escaped run')
    if (directory/'latest_checkpointed_iteration.txt').read_text().strip()!=str(step):raise ValueError('native checkpoint completion marker differs from optimizer step')
    metadata=json.loads((actor/'fsdp_config.json').read_text());world=metadata['world_size']
    if type(world) is not int or world<1 or (type(metadata.get('FSDP_version')) is not int or metadata['FSDP_version'] not in (1,2)):raise ValueError('native FSDP metadata required')
    expected={f'{kind}_world_size_{world}_rank_{rank}.pt' for kind in ('model','optim','extra_state') for rank in range(world)}
    present={p.name for p in actor.iterdir() if p.name.startswith(('model_world_size_','optim_world_size_','extra_state_world_size_'))}
    if present!=expected:raise ValueError('incomplete or conflicting native checkpoint ranks')
    for path in [actor/name for name in sorted(expected)]+[folder/'data.pt',actor/'huggingface/config.json',actor/'huggingface/tokenizer_config.json']:
        if not path.is_file() or path.stat().st_size==0 or not path.resolve().is_relative_to(root):raise ValueError('missing/unsafe native resume checkpoint file')
    evidence=evidence or TrainingEvidence(root)
    if evidence.root!=root:raise ValueError('native checkpoint evidence root differs')
    # Full resume proof includes dataloader; actor proof can feed native export.
    resume=checkpoint_inventory(folder);native=checkpoint_inventory(actor)
    evidence.event('resume_checkpoint',step,resume)
    evidence.event('training_checkpoint',step,native)
    return native


def observed_trainer_class(base,config,evidence):
    class ObservedTrainer(base):
        def fit(self):
            if not hasattr(type(self),'_officegym_native_fit'):raise ValueError('pinned effective-group native fit not installed')
            try:return self._officegym_native_fit()
            finally:
                window=getattr(self,'_officegym_group_window',None)
                if window is not None:
                    evidence.event('effective_group_sampling_end',window.step,dict(skip_reason='training_ended_before_effective_target',actor_version=window.actor,candidate_groups=window.candidates,effective_groups=window.effective,rollout_tokens=window.tokens))
                    self._officegym_group_window=None
                count=getattr(self,'_officegym_optimizer_steps',0)
                evidence.event('native_training_end',count,dict(requested_updates=config['trainer']['total_training_steps'],target_met=count==config['trainer']['total_training_steps']))
                if hasattr(self,'_dump_executor') and not self._dump_executor._shutdown:self._shutdown_dump_executor()
        def _officegym_collect_effective(self,batch,rewards,extras,metrics):
            from .native_groups import collect_native
            return collect_native(self,batch,rewards,extras,metrics,config,evidence)
        def _validate(self,*args,**kwargs):
            self._officegym_validating=True
            try:return super()._validate(*args,**kwargs)
            finally:self._officegym_validating=False
        def _get_gen_batch(self,batch):
            from .actor_version import bind_actor_metadata
            initial=config['trainer']['officegym_starting_actor_hash']
            count=getattr(self,'_officegym_optimizer_steps',0)
            expected=count if getattr(self,'_officegym_validating',False) else count+1
            if self.global_steps!=expected:raise ValueError('rollout dispatch differs from verified optimizer counter')
            if not getattr(self,'_officegym_validating',False) and getattr(self,'_officegym_group_window',None) is None:self._officegym_sampling_started=time.monotonic()
            result=super()._get_gen_batch(batch)
            column=result.non_tensor_batch['officegym'].copy()
            if not len(column):raise ValueError('nonempty native rollout dispatch required')
            for index,row in enumerate(column):column[index]=bind_actor_metadata(row,initial,count)
            result.non_tensor_batch['officegym']=column
            evidence.event('rollout_dispatch',count,dict(native_global_step=self.global_steps,actor_version=column[0]['actor_version'],rows=len(column),validation=getattr(self,'_officegym_validating',False)))
            return result
        def _update_actor(self,batch):
            validate_profile(config)
            expected=config['data']['train_batch_size']*config['actor_rollout_ref']['rollout']['n']
            count=getattr(self,'_officegym_optimizer_steps',0)
            if len(batch)!=expected or self.global_steps!=count+1:raise ValueError('native optimizer batch/counter differs from admitted single-update contract')
            initial=config['trainer'].get('officegym_starting_actor_hash')
            if initial is not None:
                from .actor_version import validate_actor_batch,actor_identity
                validate_actor_batch(batch.non_tensor_batch,initial,count,expected_rows=len(batch))
                from .native_groups import validate_effective_admission
                validate_effective_admission(batch,config,count)
            result=super()._update_actor(batch)
            def values(value):
                if isinstance(value,numbers.Real) and type(value) is not bool:return [float(value)]
                if isinstance(value,(list,tuple)):return [item for part in value for item in values(part)]
                raise ValueError('unsupported native gradient-norm evidence')
            norms=values(result.meta_info['metrics']['actor/grad_norm'])
            if not norms or any(not math.isfinite(v) or v<0 for v in norms):raise ValueError('native optimizer update lacks finite gradient proof')
            self._officegym_optimizer_steps=count+1
            version=dict(sampled_actor_step=count,sampled_actor_version=actor_identity(initial,count),per_generation_version_checks=True) if initial is not None else {}
            evidence.event('optimizer_update',count+1,dict(native_global_step=self.global_steps,gradient_norms=norms,contract='single-mini-batch-epoch',**version))
            return result
        def _save_checkpoint(self):
            if getattr(self,'_officegym_optimizer_steps',0)!=self.global_steps:raise ValueError('native checkpoint lacks verified optimizer counter')
            events=evidence.root/'training_events.jsonl'
            if events.exists() and any(row.get('kind')=='training_checkpoint' and row['optimizer_step']==self.global_steps for row in map(json.loads,events.read_text().splitlines())):
                raise ValueError('native checkpoint step already recorded; refusing overwrite')
            result=super()._save_checkpoint()
            record_saved_checkpoint(config,self.global_steps,evidence)
            return result
    return ObservedTrainer


def observed_tracking_class(base,maximum,evidence):
    class ObservedTracking(base):
        def log(self,data,step,*args,**kwargs):
            if type(step) is not int or not 0<=step<=maximum:raise ValueError('native training log step exceeds budget')
            result=super().log(data=data,step=step,*args,**kwargs)
            evidence.event('training_log',step,data)
            return result
    return ObservedTracking


def run_task(instance,config):
    root,_=validate_profile(config);checkout=Path(config.trainer.officegym_verl_checkout).resolve();hashes=verify_sources(checkout)
    import verl.trainer.main_ppo_v0 as upstream
    import verl.utils.tracking as tracking
    for module,path in ((upstream,'verl/trainer/main_ppo_v0.py'),(tracking,'verl/utils/tracking.py')):
        if Path(inspect.getfile(module)).resolve()!=checkout/path:raise ValueError('runtime imported another verl checkout')
    evidence=TrainingEvidence(root);evidence.event('backend_source_lock',0,hashes)
    from .worker_plugins import verify_loop_sources
    evidence.event('backend_rollout_source_lock',0,verify_loop_sources(checkout))
    if Path(inspect.getfile(upstream.RayPPOTrainer)).resolve()!=checkout/'verl/trainer/ppo/ray_trainer.py':raise ValueError('runtime trainer is outside locked checkout')
    trainer=observed_trainer_class(upstream.RayPPOTrainer,config,evidence)
    import verl.trainer.ppo.ray_trainer as native_trainer
    from .native_groups import instrument_fit
    trainer._officegym_native_fit=instrument_fit(checkout/'verl/trainer/ppo/ray_trainer.py',SOURCE_HASHES['verl/trainer/ppo/ray_trainer.py'],vars(native_trainer))
    evidence.event('native_fit_instrumentation',0,dict(transform='effective-groups-after-extract-reward-v1',upstream_source_sha256=SOURCE_HASHES['verl/trainer/ppo/ray_trainer.py'],instrumentation_source_sha256=hashlib.sha256(Path(inspect.getfile(instrument_fit)).read_bytes()).hexdigest()))
    original=tracking.Tracking;tracking.Tracking=observed_tracking_class(original,config.trainer.total_training_steps,evidence)
    try:
        method=extract_method(checkout/'verl/trainer/main_ppo_v0.py',SOURCE_HASHES['verl/trainer/main_ppo_v0.py'],'TaskRunner','run',dict(vars(upstream),RayPPOTrainer=trainer))
        return method(instance,config)
    finally:tracking.Tracking=original


def run_upstream(config):
    """Keep upstream validation/run_ppo, substituting only its legacy controller."""
    validate_profile(config);checkout=Path(config.trainer.officegym_verl_checkout).resolve();verify_sources(checkout)
    import ray
    import verl.trainer.main_ppo as main
    import verl.trainer.main_ppo_v0 as legacy
    if Path(inspect.getfile(main)).resolve()!=checkout/'verl/trainer/main_ppo.py':raise ValueError('runtime main is outside locked checkout')
    class TaskRunner(legacy.BaseTaskRunner):
        def run(self,config):return run_task(self,config)
    remote=ray.remote(TaskRunner);original=main.run_ppo
    def audited(config,task_runner_class):return original(config,task_runner_class=remote)
    main.run_ppo=audited
    try:return main.main.__wrapped__(config)
    finally:main.run_ppo=original

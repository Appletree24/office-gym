"""Real DataProto/pinned fit with authored rewards and backend facades; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from officegym.eval.artifacts import write_json


def worker(checkout,out):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch
    from verl import DataProto
    import verl.trainer.ppo.ray_trainer as upstream
    import verl.utils.tracking as tracking
    from officegym.training.actor_version import bind_actor_metadata
    from officegym.training.native_groups import NativeGroupWindow,instrument_fit,collect_native,validate_effective_admission
    from officegym.training.verl_checkpoint import observed_trainer_class,verify_sources,SOURCE_HASHES
    from officegym.training.evidence import TrainingEvidence
    root=Path(out).resolve();checkout=Path(checkout).resolve();verify_sources(checkout)
    initial='authored-effective-group-fixture-not-weights'
    def candidates(round_index,scores,columns=None):
        columns=dict(columns) if columns is not None else dict(uid=np.array([f'g{round_index}_{i//2}' for i in range(4)],dtype=object),officegym=np.array([bind_actor_metadata(dict(actor_version=initial,instance_dir=f'/fixture/task{i//2}'),initial,0) for i in range(4)],dtype=object))
        columns.update(officegym_actor_version=np.array([initial]*4,dtype=object),officegym_actor_update_step=np.array([0]*4,dtype=object),officegym_actor_version_verified=np.array([True]*4,dtype=object),min_global_steps=np.array([0]*4,dtype=object),max_global_steps=np.array([0]*4,dtype=object),officegym_trajectory_id=np.array([f't{round_index}_{i}' for i in range(4)],dtype=object),officegym_generated_tokens=np.array([1]*4,dtype=object),multi_modal_inputs=np.array([{} for _ in range(4)],dtype=object))
        rm=torch.zeros((4,2));rm[:,1]=torch.tensor(scores,dtype=torch.float32)
        return DataProto.from_dict(tensors=dict(attention_mask=torch.ones((4,3),dtype=torch.int64),responses=torch.tensor([[round_index*100+i,99] for i in range(4)]),response_mask=torch.tensor([[1,0]]*4),rollout_log_probs=torch.tensor([[-.5,0.]]*4),rm_scores=rm),non_tensors=columns,meta_info={'timing':{'fixture_generation':.001}})
    window=NativeGroupWindow(initial,0,2,2)
    a=candidates(1,[0,1,0,0]);b=candidates(2,[0,-1,1,1])
    selected,first=window.add(a,a.batch['rm_scores'],{});assert selected is None and first['status']=='refill'
    selected,second=window.add(b,b.batch['rm_scores'],{})
    assert second['status']=='ready' and second['candidate_groups']==4 and second['rollout_tokens']==8
    assert selected.batch['responses'][:,0].tolist()==[100,101,200,201]
    assert selected.batch['response_mask'].tolist()==[[1,0]]*4
    assert selected.batch['rollout_log_probs'].tolist()==[[-.5,0.]]*4
    assert selected.batch['rm_scores'].sum(-1).tolist()==[0,1,0,-1]
    for failure in ('reward_mismatch','stale_version','reused_trajectory','mixed_task'):
        bad=candidates(3,[0,1,0,0]);reward=bad.batch['rm_scores']
        if failure=='reward_mismatch':reward=reward+1
        elif failure=='stale_version':bad.non_tensor_batch['max_global_steps'][0]=1
        elif failure=='reused_trajectory':bad.non_tensor_batch['officegym_trajectory_id'][1]=bad.non_tensor_batch['officegym_trajectory_id'][0]
        else:bad.non_tensor_batch['officegym'][1]['instance_dir']='/foreign'
        try:NativeGroupWindow(initial,0,2,2).add(bad,reward,{})
        except ValueError:pass
        else:raise AssertionError('bad candidate batch admitted: '+failure)
    config=OmegaConf.create(dict(data=dict(train_batch_size=2),actor_rollout_ref=dict(rollout=dict(n=2,temperature=0.),actor=dict(strategy='fsdp',ppo_epochs=1,ppo_mini_batch_size=2,checkpoint=dict(async_save=False,save_contents=['model','optimizer','extra']))),algorithm=dict(adv_estimator='grpo'),global_profiler=dict(steps=None,profile_continuous_steps=False),trainer=dict(project_name='authored-probe',experiment_name='no-training',logger=[],total_training_steps=1,total_epochs=1,val_before_train=False,balance_batch=False,use_v1=False,save_freq=1,resume_mode='disable',officegym_starting_actor_hash=initial,officegym_verl_checkout=str(checkout),officegym_evidence_root=str(root),default_local_dir=str(root/'checkpoints'))))
    refill=SimpleNamespace()
    positive_root=root/'positive_collector';positive_root.mkdir()
    probe_evidence=TrainingEvidence(positive_root);metrics={}
    assert collect_native(refill,a,a.batch['rm_scores'],{},metrics,config,probe_evidence) is None
    ready=collect_native(refill,b,b.batch['rm_scores'],{},metrics,config,probe_evidence)
    assert ready is not None and refill._officegym_group_window is None
    validate_effective_admission(ready[0],config,0)
    saved=ready[0].batch['rm_scores'].clone();ready[0].batch['rm_scores'].zero_()
    try:validate_effective_admission(ready[0],config,0)
    except ValueError as error:assert 'zero-variance' in str(error)
    else:raise AssertionError('zero-variance batch bypassed optimizer admission')
    ready[0].batch['rm_scores'].copy_(saved)
    try:collect_native(refill,a,a.batch['rm_scores'],{},metrics,config,probe_evidence)
    except ValueError as error:assert 'earlier trajectory' in str(error)
    else:raise AssertionError('replayed earlier window admitted')
    class Logger:
        def __init__(self,**kwargs):pass
        def log(self,**kwargs):pass
    class Manager:
        def __init__(self):self.round=0
        def generate_sequences(self,inputs):
            self.round+=1
            return candidates(self.round,[0,0,0,0],inputs.non_tensor_batch)
    class Checkpoints:
        def __init__(self):self.versions=[];self.sleeps=0
        def update_weights(self,version):self.versions.append(version)  # facade, no weights transferred
        def sleep_replicas(self):self.sleeps+=1
    class Parent:
        _get_gen_batch=upstream.RayPPOTrainer._get_gen_batch
        _shutdown_dump_executor=upstream.RayPPOTrainer._shutdown_dump_executor
        def _load_checkpoint(self):pass
        def _start_profiling(self,*args):pass
        def _stop_profiling(self,*args):pass
        def _compute_old_log_prob(self,batch):raise ArithmeticError('zero-group reached old-logprob stage')
        def _update_actor(self,batch):raise AssertionError('optimizer must never run in zero-group probe')
    def train_batches(count):
        return [dict(fixture_prompt=torch.ones((2,1),dtype=torch.int64),officegym=np.array([dict(instance_dir=f'/fixture/task{i}',actor_version=initial) for i in range(2)],dtype=object),data_source=np.array(['officegym']*2,dtype=object)) for _ in range(count)]
    def setup(name,count,patched):
        directory=root/name;directory.mkdir();cfg=OmegaConf.create(OmegaConf.to_container(config,resolve=True));cfg.trainer.officegym_evidence_root=str(directory);cfg.trainer.default_local_dir=str(directory/'checkpoints')
        cls=observed_trainer_class(Parent,cfg,TrainingEvidence(directory))
        namespace=dict(vars(upstream),SkipManager=SimpleNamespace(init=lambda config:None,set_step=lambda step:None))
        cls._officegym_native_fit=instrument_fit(checkout/'verl/trainer/ppo/ray_trainer.py',SOURCE_HASHES['verl/trainer/ppo/ray_trainer.py'],namespace) if patched else upstream.RayPPOTrainer.fit
        obj=cls();obj.config=cfg;obj.total_training_steps=1;obj.train_dataloader=train_batches(count);obj._dump_executor=ThreadPoolExecutor(max_workers=1);obj._dump_futures=[];obj.actor_rollout_wg=SimpleNamespace();obj.async_rollout_manager=Manager();obj.checkpoint_manager=Checkpoints();obj.use_rm=False;obj.use_critic=False
        return obj,directory
    with patch.object(tracking,'Tracking',Logger),patch.object(upstream,'SkipManager',SimpleNamespace(init=lambda config:None,set_step=lambda step:None)):
        full,directory=setup('all_zero_four_batches',4,True);full.fit()
        assert full.async_rollout_manager.round==4 and full.checkpoint_manager.versions==[0]*5
        assert full.global_steps==1 and getattr(full,'_officegym_optimizer_steps',0)==0 and full._dump_executor._shutdown
        events=[json.loads(line) for line in (directory/'training_events.jsonl').read_text().splitlines()]
        sampling=[r['payload'] for r in events if r['kind']=='effective_group_sampling']
        assert [r['candidate_groups'] for r in sampling]==[2,4,6,8]
        assert [r['rollout_tokens'] for r in sampling]==[4,8,12,16]
        assert sampling[-1]['status']=='insufficient_effective_groups'
        write_json(directory/'observed_events.json',events)
        assert not any(r['kind']=='optimizer_update' for r in events)
        short,shortdir=setup('data_exhaustion',2,True);short.fit();assert short._officegym_group_window is None and short._dump_executor._shutdown
        ending=[json.loads(line) for line in (shortdir/'training_events.jsonl').read_text().splitlines()]
        assert any(row['kind']=='effective_group_sampling_end' and row['payload']['candidate_groups']==4 for row in ending)
        old,olddir=setup('unmodified_fit_counterexample',1,False)
        try:old.fit()
        except ArithmeticError as error:assert 'old-logprob' in str(error)
        else:raise AssertionError('unmodified fit unexpectedly filtered zero groups')
    write_json(root/'report.json',dict(passed=True,kind='authored-reward-real-dataproto-pinned-fit-control-flow',actual_optimizer_updates=0,model_generations=0,physical_weight_sync_verified=False,positive_selected_response_ids=selected.batch['responses'][:,0].tolist(),positive_sampling=second,zero_candidate_groups=8,zero_rollout_tokens=16,zero_generation_batches=4,version_wakeup_requests=full.checkpoint_manager.versions,data_exhaustion_recorded=True,replay_across_windows_rejected=True,unmodified_fit_reaches_logprob=True,source_sha256=SOURCE_HASHES['verl/trainer/ppo/ray_trainer.py']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkout',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.checkout,args.out)
    else:
        from officegym.training.process_tree import execute_tree_bounded
        root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
        (root/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
        for name in ('native_groups.py','verl_checkpoint.py','verl_loop.py'):(root/('executed_'+name)).write_bytes((Path('src/officegym/training')/name).read_bytes())
        with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
            status=execute_tree_bounded([sys.executable,str(Path(__file__).resolve()),'--worker','--checkout',str(Path(args.checkout).resolve()),'--out',str(root)],stdout,stderr,180,evidence_dir=root/'ownership')
        if status['exit_code'] or status['wall_budget_exhausted']:raise SystemExit(1)
        print((root/'report.json').read_text())

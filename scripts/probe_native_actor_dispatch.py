"""Pinned controller/DataProto dispatch probe; update counters are authored fixtures."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from officegym.eval.artifacts import write_json


def worker(checkout,out):
    import numpy as np
    import torch
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from officegym.training.verl_checkpoint import observed_trainer_class,verify_sources
    from officegym.training.worker_plugins import verify_loop_sources
    from officegym.training.evidence import TrainingEvidence
    from officegym.training.actor_version import actor_identity
    root=Path(out).resolve();checkout=Path(checkout).resolve()
    checkpoint_sources=verify_sources(checkout);rollout_sources=verify_loop_sources(checkout)
    assert Path(inspect.getfile(RayPPOTrainer)).resolve()==checkout/'verl/trainer/ppo/ray_trainer.py'
    initial='authored-controller-fixture-not-a-checkpoint'
    config=dict(data={'train_batch_size':2},trainer=dict(total_training_steps=2,use_v1=False,save_freq=1,officegym_evidence_root=str(root),default_local_dir=str(root/'checkpoints'),officegym_verl_checkout=str(checkout),officegym_starting_actor_hash=initial,resume_mode='disable'),actor_rollout_ref=dict(rollout={'n':2},actor=dict(strategy='fsdp',ppo_epochs=1,ppo_mini_batch_size=2,checkpoint=dict(async_save=False,save_contents=['model','optimizer','extra']))))
    def batch():
        metadata=[dict(instance_dir='/authored/task/'+str(i),actor_version=initial) for i in range(2)]
        result=DataProto.from_dict(tensors={'input_ids':torch.ones((2,3),dtype=torch.int64)},non_tensors={'officegym':np.array(metadata,dtype=object),'data_source':np.array(['officegym']*2,dtype=object)})
        return result,metadata
    class Parent:
        _get_gen_batch=RayPPOTrainer._get_gen_batch
        def _validate(self):return self._get_gen_batch(batch()[0])
        def _update_actor(self,batch):raise AssertionError('no optimizer execution permitted in dispatch probe')
    controller=observed_trainer_class(Parent,config,TrainingEvidence(root))()
    versions=[]
    for step in (0,1):
        controller._officegym_optimizer_steps=step  # Authored prior-update fixture, not trained weights.
        controller.global_steps=step+1
        source,metadata=batch();dispatch=controller._get_gen_batch(source)
        assert all(row==dict(instance_dir='/authored/task/'+str(i),actor_version=initial) for i,row in enumerate(metadata))
        assert all(row['actor_update_step']==step and row['actor_version']==actor_identity(initial,step) for row in dispatch.non_tensor_batch['officegym'])
        repeated=dispatch.repeat(repeat_times=2,interleave=True)
        assert len(repeated)==4 and all(row['actor_update_step']==step for row in repeated.non_tensor_batch['officegym'])
        versions.append(repeated.non_tensor_batch['officegym'][0]['actor_version'])
        # Initial and post-update validation use completed-step count, not next-update index.
        controller.global_steps=step;validated=controller._validate()
        assert all(row['actor_update_step']==step for row in validated.non_tensor_batch['officegym'])
        assert controller._officegym_validating is False
    assert versions[0]!=versions[1]
    controller.global_steps=99
    try:controller._get_gen_batch(batch()[0])
    except ValueError as error:assert 'optimizer counter' in str(error)
    else:raise AssertionError('counter mismatch admitted')
    controller.global_steps=2
    columns=repeated.non_tensor_batch
    columns.update(officegym_actor_version=np.array([versions[1]]*4,dtype=object),officegym_actor_update_step=np.array([1]*4,dtype=object),officegym_actor_version_verified=np.array([True]*4,dtype=object),min_global_steps=np.array([1]*4,dtype=object),max_global_steps=np.array([1,1,0,1],dtype=object))
    try:controller._update_actor(repeated)
    except ValueError as error:assert 'stale or mixed' in str(error)
    else:raise AssertionError('stale optimizer row admitted')
    events=[json.loads(line) for line in (root/'training_events.jsonl').read_text().splitlines()]
    assert len(events)==4 and all(event['kind']=='rollout_dispatch' for event in events)
    write_json(root/'report.json',dict(passed=True,kind='authored-update-counter-real-dataproto-controller-probe',actual_optimizer_updates=0,model_generations=0,physical_weight_sync_verified=False,simulated_completed_update_steps=[0,1],actor_versions=versions,source_hashes={**checkpoint_sources,**rollout_sources},local_source_hashes={name:hashlib.sha256((Path('src/officegym/training')/name).read_bytes()).hexdigest() for name in ('actor_version.py','verl_checkpoint.py')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkout',required=True);parser.add_argument('--out',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.checkout,args.out)
    else:
        from officegym.training.process_tree import execute_tree_bounded
        root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
        (root/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
        with (root/'stdout.log').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
            status=execute_tree_bounded([sys.executable,str(Path(__file__).resolve()),'--worker','--checkout',str(Path(args.checkout).resolve()),'--out',str(root)],stdout,stderr,180,evidence_dir=root/'ownership')
        if status['exit_code'] or status['wall_budget_exhausted']:raise SystemExit(1)
        print((root/'report.json').read_text())

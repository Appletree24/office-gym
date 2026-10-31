"""Real native-engine diagnostic optimizer/save and fresh-process reload checks.

The advantage is authored; these checkpoints are not admitted training exports.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from officegym.eval.artifacts import write_json


def state_fingerprint(value):
    import numpy as np
    import torch
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(str((item.dtype,tuple(item.shape))).encode())
            item=item.detach()
            if hasattr(item,'to_local'):item=item.to_local()
            digest.update(item.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,np.ndarray):digest.update(str((item.dtype,item.shape)).encode());digest.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=repr):digest.update(repr(key).encode());visit(item[key])
        elif isinstance(item,(list,tuple)):
            for part in item:visit(part)
        else:digest.update(repr(item).encode())
    visit(value);return digest.hexdigest()


def engine_state(engine):
    import numpy as np
    import torch
    return dict(adapter=state_fingerprint({n:p for n,p in engine.module.named_parameters() if p.requires_grad}),optimizer=state_fingerprint(engine.optimizer.state_dict()),scheduler=state_fingerprint(engine.lr_scheduler.state_dict()),rng=state_fingerprint(dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(0))))


def advance_and_save(engine,infer,root):
    from .evidence import checkpoint_inventory
    root=Path(root);before=engine_state(engine)
    norm=engine.optimizer_step()
    if not math.isfinite(norm) or norm<=0:raise ValueError('diagnostic update lacks finite nonzero native gradient norm')
    lr=engine.lr_scheduler_step();engine.optimizer_zero_grad()
    after=engine_state(engine)
    if after['adapter']==before['adapter'] or after['optimizer']==before['optimizer']:raise ValueError('native optimizer did not change adapter/optimizer state')
    checkpoint=root/'native_checkpoint_step_1'
    engine.save_checkpoint(str(checkpoint),global_step=1)
    saved=engine_state(engine)
    if saved!=after:raise ValueError('native save changed model/optimizer/scheduler/RNG state')
    actor=infer();reference=infer(True)
    write_json(root/'updated_logprobs.json',dict(actor=actor.tolist(),reference=reference.tolist()))
    result=dict(kind='native-engine-authored-advantage-checkpoint-not-formal-training',worker_pid=os.getpid(),optimizer_updates=1,authored_advantage=True,gradient_norm=norm,learning_rate_after_scheduler=lr,state=saved,checkpoint=checkpoint_inventory(checkpoint),reference_is_admitted_sft=False)
    write_json(root/'native_checkpoint_result.json',result)
    return result



def continue_update(engine,data,infer,root):
    import torch
    from verl.workers.config.actor import ActorConfig
    from verl.workers.utils.losses import ppo_loss
    from verl.utils import tensordict_utils as tu
    old=infer();fixed=infer(True);before=engine_state(engine)
    training=data.clone();training['old_log_probs']=old;training['advantages']=torch.ones_like(old)
    tu.assign_non_tensor(training,global_batch_size=2,return_model_output=False)
    config=ActorConfig(strategy='fsdp2',rollout_n=2,ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=1,use_torch_compile=False,use_kl_loss=False,loss_agg_mode='token-mean')
    def loss(model_output,data,dp_group):return ppo_loss(config,model_output,data,dp_group)
    with engine.train_mode():result=engine.train_batch(training,loss)
    norm=result['metrics']['grad_norm']
    if not math.isfinite(float(norm)) or norm<=0:raise ValueError('continued native update lacks finite positive gradient')
    engine.lr_scheduler_step();after=engine_state(engine)
    if before['adapter']==after['adapter'] or before['optimizer']==after['optimizer']:raise ValueError('continued native update did not change state')
    actor=infer();reference=infer(True)
    if not torch.equal(fixed,reference):raise ValueError('continued update changed disabled-adapter reference')
    root=Path(root)
    write_json(root/'continuation_vectors.json',dict(actor=actor.tolist(),reference=reference.tolist()))
    value=dict(passed=True,authored_advantage=True,state=after,loss_fingerprint=state_fingerprint(result['loss']),gradient_norm=float(norm),optimizer_updates_in_this_call=1,formal_training=False)
    write_json(root/'continuation_result.json',value);return value


def reload_and_verify(engine,infer,root,source_root,data=None,continuation=False):
    import torch
    from .evidence import checkpoint_inventory
    source=Path(source_root);root=Path(root)
    prior=json.loads((source/'native_checkpoint_result.json').read_text())
    if prior['optimizer_updates']!=1 or not prior['authored_advantage'] or prior['worker_pid']==os.getpid():raise ValueError('separate diagnostic writer process required')
    checkpoint=source/'native_checkpoint_step_1'
    if checkpoint_inventory(checkpoint)!=prior['checkpoint']:raise ValueError('native checkpoint files changed before reload')
    before=engine_state(engine)
    if before['adapter']==prior['state']['adapter']:raise ValueError('fresh model already contains updated adapter')
    engine.load_checkpoint(str(checkpoint),del_local_after_load=False)
    restored=engine_state(engine)
    if restored!=prior['state']:raise ValueError('fresh-process native model/optimizer/scheduler/RNG restoration differs')
    vectors=json.loads((source/'updated_logprobs.json').read_text())
    actor=infer();reference=infer(True)
    if not torch.equal(actor,torch.tensor(vectors['actor'])) or not torch.equal(reference,torch.tensor(vectors['reference'])):raise ValueError('fresh-process checkpoint logprobs differ')
    if checkpoint_inventory(checkpoint)!=prior['checkpoint']:raise ValueError('reload deleted or changed source checkpoint')
    continuation_verified=False
    if continuation:
        value=continue_update(engine,data,infer,root)
        expected=json.loads((source/'continuation_result.json').read_text())
        if value!=expected:raise ValueError('resumed native optimizer continuation differs from uninterrupted run')
        if (root/'continuation_vectors.json').read_bytes()!=(source/'continuation_vectors.json').read_bytes():raise ValueError('resumed step-two actor/reference vectors differ')
        continuation_verified=True
    write_json(root/'report.json',dict(passed=True,continued_update_bitwise_equal=continuation_verified,kind='fresh-process-native-engine-checkpoint-reload-not-formal-training',writer_pid=prior['worker_pid'],reader_pid=os.getpid(),source_checkpoint_sha256=prior['checkpoint']['checkpoint_files_sha256'],restored_state=restored,model_optimizer_scheduler_rng_equal=True,actor_reference_vectors_bitwise_equal=True,optimizer_updates=int(continuation),source_diagnostic_optimizer_updates=2 if continuation else 1,model_generations=0,teacher_samples=0,reference_is_admitted_sft=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch-root',required=True);parser.add_argument('--source',required=True);parser.add_argument('--out',required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--continuation',action='store_true');args=parser.parse_args()
    from .native_inference_probe import run
    print(json.dumps(run(args.fetch_root,args.source,args.out,backend=('native-engine-resume' if args.resume else 'native-engine-update')+('-continuation' if args.continuation else ''))))

"""SFT optimizer-call evidence, distinct from checkpoint files and log counters."""
import json
import math
from pathlib import Path

CONTRACT='sft-single-optimizer-step-v1'
FIELDS={'contract','gradient_tensors','gradient_parameters','trainable_parameters','gradient_l2_norm'}


def validate_gradient(proof):
    if not isinstance(proof,dict) or set(proof)!=FIELDS or proof.get('contract')!=CONTRACT:raise ValueError('complete SFT gradient evidence required')
    for name in ('gradient_tensors','gradient_parameters','trainable_parameters'):
        if type(proof[name]) is not int or proof[name]<1:raise ValueError('positive SFT gradient counts required')
    if proof['gradient_parameters']>proof['trainable_parameters'] or proof['gradient_tensors']>proof['gradient_parameters']:raise ValueError('inconsistent SFT gradient counts')
    value=proof['gradient_l2_norm']
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative SFT gradient norm required')
    return proof


def gradient_proof(model):
    import torch
    count=parameters=trainable=0;norm_squared=0.
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        trainable+=parameter.numel()
        if parameter.grad is None:continue
        grad=parameter.grad.detach()
        if grad.is_sparse or not torch.isfinite(grad).all():raise ValueError('nonfinite or unsupported SFT gradient')
        norm_squared+=grad.double().square().sum().item();count+=1;parameters+=parameter.numel()
    return validate_gradient(dict(contract=CONTRACT,gradient_tensors=count,gradient_parameters=parameters,trainable_parameters=trainable,gradient_l2_norm=math.sqrt(norm_squared)))


class UpdateRecorder:
    def __init__(self,evidence):self.evidence=evidence;self.pending=None;self.completed=0
    def attempt(self,step,proof):
        validate_gradient(proof)
        if type(step) is not int or step!=self.completed+1 or self.pending is not None:raise ValueError('SFT update attempt sequence differs')
        self.evidence.event('sft_optimizer_attempt',step,proof);self.pending=(step,dict(proof))
    def finish(self,step,*,skipped):
        if self.pending is None or step!=self.pending[0] or type(skipped) is not bool:raise ValueError('unmatched SFT optimizer completion')
        proof=dict(self.pending[1],optimizer_step_was_skipped=skipped)
        if skipped:
            self.evidence.event('sft_optimizer_skipped',step,proof)
            raise ValueError('SFT optimizer step was skipped; attempt retained')
        self.evidence.event('sft_optimizer_update',step,proof);self.completed=step;self.pending=None
    def step_end(self,step):
        if self.pending is not None or step!=self.completed:raise ValueError('SFT global step differs from completed optimizer calls')


def make_update_callback(evidence,skip_checker):
    """Canonical callback shared by production SFT and native integration probes."""
    from transformers import TrainerCallback
    updates=UpdateRecorder(evidence)
    class Callback(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            try:proof=gradient_proof(model)
            except ValueError:
                evidence.event('sft_gradient_rejected',state.global_step+1,{'reason':'nonfinite, unsupported or missing gradient evidence'})
                raise
            updates.attempt(state.global_step+1,proof)
        def on_optimizer_step(self,args,state,control,**kwargs):
            updates.finish(state.global_step+1,skipped=skip_checker())
        def on_step_end(self,args,state,control,**kwargs):updates.step_end(state.global_step)
    return Callback()


def verify_updates(events_path,prefix_bytes,optimizer_step):
    raw=Path(events_path).read_bytes()
    if type(prefix_bytes) is not int or not 0<prefix_bytes<=len(raw) or raw[prefix_bytes-1:prefix_bytes]!=b'\n' or type(optimizer_step) is not int or optimizer_step<1:raise ValueError('valid SFT checkpoint evidence prefix required')
    pending=None;completed=0
    for line in raw[:prefix_bytes].splitlines():
        record=json.loads(line);kind=record.get('kind');step=record.get('optimizer_step');proof=record.get('payload',{})
        if kind in ('training_failed','sft_optimizer_skipped','sft_gradient_rejected'):raise ValueError('failed or skipped SFT update before checkpoint')
        if kind=='sft_optimizer_attempt':
            validate_gradient(proof)
            if type(step) is not int or step!=completed+1 or pending is not None:raise ValueError('duplicate or out-of-order SFT optimizer attempt')
            pending=(step,proof)
        elif kind=='sft_optimizer_update':
            if pending is None or type(step) is not int or step!=pending[0] or proof!=dict(pending[1],optimizer_step_was_skipped=False) or proof.get('optimizer_step_was_skipped') is not False:raise ValueError('unmatched SFT optimizer update')
            completed=step;pending=None
    if pending is not None or completed!=optimizer_step:raise ValueError('SFT checkpoint lacks complete optimizer-call evidence')
    return completed

"""Bounded native DataProto effective-group buffering and pinned fit insertion."""
import ast
from collections import Counter
import hashlib
import math
from pathlib import Path
import time
from .actor_version import actor_identity,validate_actor_batch,validate_actor_metadata


class NativeGroupWindow:
    def __init__(self,initial,step,target,size,multiplier=4):
        if any(type(v) is not int or v<1 for v in (target,size,multiplier)) or size<2 or multiplier>4:
            raise ValueError('bounded complete effective-group profile required')
        self.initial=initial;self.step=step;self.actor=actor_identity(initial,step)
        self.target=target;self.size=size;self.cap=target*multiplier
        self.parts=[];self.candidates=0;self.effective=0;self.tokens=0;self.rounds=0;self.tool_calls_known=0;self.tool_trajectories_unknown=0
        self.histogram=Counter();self.seen=set();self.uids=set();self.started=time.monotonic()

    def add(self,batch,rewards,extras):
        if extras:raise ValueError('native OfficeGym sampling requires its terminal-only reward path')
        if len(batch)!=self.target*self.size:raise ValueError('complete native candidate batch required')
        if 'rm_scores' not in batch.batch or not rewards.equal(batch.batch['rm_scores']):raise ValueError('sampling reward differs from trusted native terminal scores')
        if self.candidates>=self.cap:raise ValueError('effective-group window already exhausted')
        validate_actor_batch(batch.non_tensor_batch,self.initial,self.step,expected_rows=len(batch))
        scores=rewards.sum(-1).detach().cpu().tolist()
        columns=batch.non_tensor_batch;groups={}
        for index,uid in enumerate(columns['uid']):
            if not isinstance(uid,str) or not uid:raise ValueError('native group identity required')
            groups.setdefault(uid,[]).append(index)
        if len(groups)!=self.target or any(len(rows)!=self.size for rows in groups.values()):raise ValueError('native uid group is incomplete')
        if self.uids & groups.keys():raise ValueError('native refill reused a group identity')
        self.uids.update(groups)
        # Validate the entire generated batch before accepting any group.
        trajectories=[]
        tool_counts=columns.get('officegym_tool_calls',[None]*len(batch))
        if len(tool_counts)!=len(batch) or any(value is not None and (type(value) is not int or value<0) for value in tool_counts):raise ValueError('invalid native tool-call usage')
        for index,score in enumerate(scores):
            if validate_actor_metadata(columns['officegym'][index],self.initial)!=self.step:raise ValueError('native refill metadata actor changed')
            if type(score) not in (int,float) or not math.isfinite(score) or score not in (-1,0,1):raise ValueError('native terminal rewards must be finite 1/0/-1')
            trajectory=columns['officegym_trajectory_id'][index]
            tokens=columns['officegym_generated_tokens'][index]
            if not isinstance(trajectory,str) or not trajectory or trajectory in self.seen or trajectory in trajectories:raise ValueError('native refill reused a trajectory')
            if type(tokens) is not int or tokens<0:raise ValueError('native generated-token usage required')
            trajectories.append(trajectory)
        self.seen.update(trajectories)
        for uid,indices in groups.items():
            if len({columns['officegym'][i]['instance_dir'] for i in indices})!=1:raise ValueError('native group mixed initial worlds')
            self.candidates+=1
            values=[scores[i] for i in indices]
            for i in indices:
                self.tokens+=columns['officegym_generated_tokens'][i]
                if tool_counts[i] is None:self.tool_trajectories_unknown+=1
                else:self.tool_calls_known+=tool_counts[i]
                self.histogram[str(float(scores[i]))]+=1
            if len(set(values))>1:
                self.effective+=1
                if len(self.parts)<self.target:
                    part=batch.select_idxs(indices)
                    # Candidate-local lengths/timings cannot be concatenated as
                    # if they described the selected optimizer batch.
                    part.meta_info={}
                    self.parts.append(part)
        self.rounds+=1
        ready=len(self.parts)==self.target
        exhausted=self.candidates==self.cap
        status='ready' if ready else 'insufficient_effective_groups' if exhausted else 'refill'
        stats=dict(status=status,actor_version=self.actor,actor_update_step=self.step,candidate_groups=self.candidates,effective_groups=self.effective,selected_groups=len(self.parts),effective_group_ratio=self.effective/self.candidates,rollout_tokens=self.tokens,tool_calls_known=self.tool_calls_known,tool_trajectories_unknown=self.tool_trajectories_unknown,tool_calls_exact=self.tool_trajectories_unknown==0,refill_attempts=max(0,self.candidates-self.target),refill_rounds=self.rounds-1,generation_batches=self.rounds,reward_histogram=dict(self.histogram),window_seconds=time.monotonic()-self.started)
        if not ready:return None,stats
        from verl import DataProto
        selected=DataProto.concat(self.parts)
        selected.meta_info={key:value for key,value in batch.meta_info.items() if key not in ('global_token_num','images_seqlens','metrics')}
        selected.meta_info['global_token_num']=selected.batch['attention_mask'].sum(-1).tolist()
        selected.meta_info['images_seqlens']=[]
        # Rewards remain the trusted original per-token scores of selected rows.
        return selected,stats


def collect_native(trainer,batch,rewards,extras,metrics,config,evidence):
    count=getattr(trainer,'_officegym_optimizer_steps',0)
    window=getattr(trainer,'_officegym_group_window',None)
    if window is None:
        window=NativeGroupWindow(config['trainer']['officegym_starting_actor_hash'],count,config['data']['train_batch_size'],config['actor_rollout_ref']['rollout']['n'])
        window.started=getattr(trainer,'_officegym_sampling_started',window.started)
        trainer._officegym_group_window=window
    if window.step!=count:raise ValueError('actor changed during effective-group refill')
    seen=getattr(trainer,'_officegym_seen_trajectories',set())
    trajectories=set(batch.non_tensor_batch['officegym_trajectory_id'])
    if seen & trajectories:raise ValueError('native refill replayed an earlier trajectory')
    selected,stats=window.add(batch,rewards,extras)
    trainer._officegym_seen_trajectories=seen|trajectories
    evidence.event('effective_group_sampling',count,stats)
    if stats['status']!='refill':trainer._officegym_group_window=None
    if selected is None:return None
    if config['trainer'].get('balance_batch',False):trainer._balance_batch(selected,metrics=metrics)
    selected.meta_info['global_token_num']=selected.batch['attention_mask'].sum(-1).tolist()
    selected.meta_info['officegym_effective_group_admission']=dict(actor_update_step=count,target_groups=window.target,group_size=window.size,candidate_groups=stats['candidate_groups'],trajectory_ids=sorted(selected.non_tensor_batch['officegym_trajectory_id'].tolist()))
    metrics.update({'officegym_sampling/'+key:value for key,value in stats.items() if type(value) in (int,float)})
    return selected,selected.batch['rm_scores'],{}


def validate_effective_admission(batch,config,step):
    admission=batch.meta_info.get('officegym_effective_group_admission',{})
    target=config['data']['train_batch_size'];size=config['actor_rollout_ref']['rollout']['n']
    if any(type(admission.get(key)) is not int or admission[key]!=value for key,value in (('actor_update_step',step),('target_groups',target),('group_size',size))):raise ValueError('native optimizer lacks current effective-group admission')
    candidates=admission.get('candidate_groups')
    if type(candidates) is not int or not target<=candidates<=target*4:raise ValueError('native effective-group candidate bound differs')
    trajectories=batch.non_tensor_batch['officegym_trajectory_id'].tolist()
    if len(set(trajectories))!=len(batch) or sorted(trajectories)!=admission.get('trajectory_ids'):raise ValueError('native effective-group rows changed after admission')
    groups={}
    scores=batch.batch['rm_scores'].sum(-1).detach().cpu().tolist()
    for uid,score in zip(batch.non_tensor_batch['uid'],scores):
        if not math.isfinite(score) or score not in (-1,0,1):raise ValueError('invalid native admitted terminal reward')
        groups.setdefault(uid,[]).append(score)
    if len(groups)!=target or any(len(values)!=size or len(set(values))<2 for values in groups.values()):raise ValueError('native optimizer received incomplete or zero-variance groups')


def instrument_fit(path,expected_sha256,namespace):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('pinned native fit source changed')
    tree=ast.parse(raw,filename=str(path))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RayPPOTrainer')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='fit')
    if method.decorator_list:raise ValueError('unexpected native fit decorator')
    insertion=ast.parse('''
_officegym_selected = self._officegym_collect_effective(batch, reward_tensor, reward_extra_infos_dict, metrics)
if _officegym_selected is None:
    self.checkpoint_manager.update_weights(self.global_steps - 1)
    self._stop_profiling(curr_step_profile)
    prev_step_profile = False
    continue
batch, reward_tensor, reward_extra_infos_dict = _officegym_selected
''').body
    class Insert(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='extract_reward':
                names=[item.id for item in node.targets[0].elts] if isinstance(node.targets[0],ast.Tuple) else []
                if names!=['reward_tensor','reward_extra_infos_dict']:raise ValueError('native reward extraction shape changed')
                self.count+=1
                return [node,*insertion]
            return node
    transform=Insert();method=transform.visit(method)
    if transform.count!=1:raise ValueError('native fit reward insertion must be unique')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]));scope=dict(namespace)
    exec(compile(module,str(path),'exec'),scope)
    return scope['fit']

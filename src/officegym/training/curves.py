"""Reconstruct native GRPO operational curves without inventing evaluation data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .checkpoints import verify_grpo_updates
from officegym.eval.artifacts import write_json


def summarize(events_path):
    path=Path(events_path);raw=path.read_bytes();rows=[];offset=0
    for line in raw.splitlines(keepends=True):
        offset+=len(line)
        if line.strip():rows.append((offset,json.loads(line)))
    sampling=[];updates=[];checkpoints=[];window=None;window_number=0
    tokens=0;candidates=0;effective=0;trained=0;context=0;pending=None
    tool_calls=0;unknown_tool_trajectories=0;tool_metadata_complete=True
    for index,(end,row) in enumerate(rows):
        kind=row['kind'];step=row['optimizer_step'];payload=row['payload']
        if kind=='effective_group_sampling':
            status=payload.get('status')
            if status not in ('refill','ready','insufficient_effective_groups'):raise ValueError('unknown native sampling status')
            values={key:payload.get(key) for key in ('rollout_tokens','candidate_groups','effective_groups')}
            if any(type(value) is not int or value<0 for value in values.values()) or values['effective_groups']>values['candidate_groups']:
                raise ValueError('invalid cumulative native sampling costs')
            if window is None:
                window_number+=1;window=dict(step=step,actor=payload.get('actor_version'),tool_counts=None,rollout_tokens=0,candidate_groups=0,effective_groups=0)
            if step!=window['step'] or payload.get('actor_version')!=window['actor'] or any(values[key]<window[key] for key in values):
                raise ValueError('sampling window actor/step changed or counters regressed')
            increments={key:values[key]-window[key] for key in values}
            if increments['candidate_groups']<1 or increments['effective_groups']>increments['candidate_groups']:
                raise ValueError('sampling group increments are inconsistent')
            tool_keys=('tool_calls_known','tool_trajectories_unknown')
            present=any(key in payload for key in tool_keys)
            batch_tools=None;batch_tool_unknown=None
            if present:
                current=tuple(payload.get(key) for key in tool_keys)
                if any(type(value) is not int or value<0 for value in current):raise ValueError('invalid sampling tool-call counters')
                previous=window['tool_counts']
                if previous is None:
                    if window['candidate_groups']!=0:raise ValueError('tool counters appeared mid-window')
                    previous=(0,0)
                if any(a<b for a,b in zip(current,previous)):raise ValueError('sampling tool counters regressed')
                batch_tools=current[0]-previous[0];tool_calls+=batch_tools
                batch_tool_unknown=current[1]-previous[1];unknown_tool_trajectories+=batch_tool_unknown;window['tool_counts']=current
            else:
                if window['tool_counts'] is not None:raise ValueError('sampling tool counters disappeared mid-window')
                tool_metadata_complete=False
            tokens+=increments['rollout_tokens'];candidates+=increments['candidate_groups'];effective+=increments['effective_groups']
            sampling.append(dict(event_index=index,optimizer_step=step,window=window_number,status=status,actor_version=window['actor'],batch_tool_calls_known=batch_tools,batch_tool_trajectories_unknown=batch_tool_unknown,cumulative_reported_tool_calls_known=tool_calls,batch_generated_tokens=increments['rollout_tokens'],batch_candidate_groups=increments['candidate_groups'],batch_effective_groups=increments['effective_groups'],cumulative_reported_generated_tokens=tokens,cumulative_candidate_groups=candidates,cumulative_effective_groups=effective))
            window.update(values)
            if status!='refill':window=None
        elif kind=='effective_group_sampling_end':
            # This is a restatement of an unfinished window, not another generation.
            if window is not None:
                if step!=window['step'] or payload.get('actor_version')!=window['actor'] or any(payload.get(key)!=window[key] for key in ('rollout_tokens','candidate_groups','effective_groups')):
                    raise ValueError('sampling end cannot be reconciled with recorded window')
                window=None
            elif any(payload.get(key,0)!=0 for key in ('rollout_tokens','candidate_groups','effective_groups')):
                raise ValueError('orphan sampling end lacks recorded source batches')
        elif kind=='optimizer_update_attempt':
            if pending is not None:raise ValueError('ambiguous repeated optimizer attempt')
            pending=dict(optimizer_step=step,event_index=index,payload=payload)
        elif kind=='optimizer_update':
            verify_grpo_updates(path,end,step)
            if pending is None or pending['optimizer_step']!=step:raise ValueError('verified update lacks matching attempt')
            trained+=payload['actor_loss_tokens'];context+=payload['forward_context_tokens']
            updates.append(dict(optimizer_step=step,event_index=index,selected_tool_calls_known=payload.get('selected_tool_calls_known'),selected_tool_trajectories_unknown=payload.get('selected_tool_trajectories_unknown'),actor_loss_tokens=payload['actor_loss_tokens'],forward_context_tokens=payload['forward_context_tokens'],cumulative_actor_loss_tokens=trained,cumulative_forward_context_tokens=context,cumulative_reported_generated_tokens=tokens,cumulative_candidate_groups=candidates,cumulative_effective_groups=effective,reference_kl=payload['native_actor_metrics']['actor/kl_loss'],gradient_norm_max=max(payload['gradient_norms']),sampled_actor_version=payload['sampled_actor_version']))
            pending=None
        elif kind in ('training_checkpoint','final_adapter'):
            checkpoints.append(dict(optimizer_step=step,event_index=index,path=payload.get('path'),checkpoint_files_sha256=payload.get('checkpoint_files_sha256')))
    admission=bool(rows and rows[0][1].get('kind')=='training_admission')
    if admission and rows[0][1]['payload'].get('stage')!='grpo':raise ValueError('native GRPO curve input cannot be a different training stage')
    return dict(format='officegym-native-training-curves-v1',kind='operational_evidence_not_evaluation_results',events_sha256=hashlib.sha256(raw).hexdigest(),formal_admission_present_but_not_verified=admission,sampling_batches=sampling,updates=updates,checkpoints=checkpoints,totals=dict(reported_sampling_tool_calls_known=tool_calls,reported_sampling_tool_trajectories_unknown=unknown_tool_trajectories if tool_metadata_complete else None,reported_sampling_tool_calls_exact=bool(sampling) and tool_metadata_complete and unknown_tool_trajectories==0,reported_sampling_tokens=tokens,candidate_groups=candidates,effective_groups=effective,actor_loss_tokens=trained,forward_context_tokens=context,protocol_verified_updates=len(updates)),unresolved_optimizer_attempt=pending,open_sampling_window=window,ledger_usage=None,costs_not_recorded_here=['gpu_hours','api_cost_usd','tool_calls_outside_recorded_sampling','generation_cost_outside_recorded_sampling'],evaluation_success_rates=None)


def export(events_path,out,*,usage_path=None):
    result=summarize(events_path)
    if usage_path is not None:
        path=Path(usage_path);usage=json.loads(path.read_text())
        for key in ('actual_tokens_lower_bound','actual_tokens_upper_bound','observed_tokens_known','reserved_tokens'):
            if type(usage.get(key)) is not int or usage[key]<0:raise ValueError('invalid ledger usage interval')
        lower=usage['actual_tokens_lower_bound'];upper=usage['actual_tokens_upper_bound']
        if lower!=usage['observed_tokens_known'] or not lower<=upper<=usage['reserved_tokens'] or result['totals']['reported_sampling_tokens']>lower:
            raise ValueError('sampling costs cannot be reconciled with ledger observations')
        result['costs_not_recorded_here'].remove('generation_cost_outside_recorded_sampling')
        result['ledger_usage']=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),usage=usage)
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    for name in ('updates','sampling_batches','checkpoints'):
        rows=result[name]
        with (root/(name+'.csv')).open('w',newline='') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--events',required=True);parser.add_argument('--out',required=True);parser.add_argument('--usage');args=parser.parse_args()
    result=export(args.events,args.out,usage_path=args.usage)
    print(json.dumps(result['totals']))

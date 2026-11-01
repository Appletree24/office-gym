"""Concurrent conditional simple/full recovery from one controller-owned prefix."""
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
from officegym.env.service import serve
from .artifacts import write_json
from .fault_checkpoint import branch_checkpoint,verify_checkpoint,digest


def recover_pair(root,config,task,private,versions,origin,source_fault,software_fixture,max_generated_tokens):
    from .fault_episode import worker,finalize_episode
    root=Path(root);source=root/'committed_checkpoint';checkpoint=verify_checkpoint(source);deadline=origin+config['budget']['wall_seconds'];context=multiprocessing.get_context('spawn')
    prefix_records=[json.loads(line) for line in (source/'harness.jsonl').read_text().splitlines()]
    prefix=dict(generated_tokens=sum(r.get('output_tokens',0) if r['kind']=='model_response' else r.get('summary_output_tokens',0) if r['kind']=='context_selection' else 0 for r in prefix_records),input_tokens=sum(r.get('input_tokens',0) if r['kind']=='model_response' else r.get('summary_input_tokens',0) if r['kind']=='context_selection' else 0 for r in prefix_records))
    branches={};processes=[];channels={};events={mode:[] for mode in ('simple','full')};completed={};faults={}
    with tempfile.TemporaryDirectory(prefix='officegym-fault-pair-') as temporary:
        try:
            for mode in ('simple','full'):
                branch=root/('recovery_'+mode);binding=branch_checkpoint(source,branch);branches[mode]=branch
                for name in ('task.json','private_spec.json','event_rules.json','initial_world.sqlite','tokenizer_manifest.json'):
                    shutil.copyfile(root/name,branch/name)
                write_json(branch/'source_usage.json',prefix)
                write_json(branch/'policy_isolation.json',dict(version='policy-chroot-v1',verified=False,status='not_started'))
                write_json(branch/'resolved_eval_config.json',dict(config,mode=mode))
                socket=Path(temporary)/(mode+'.sock');server=context.Process(target=serve,args=(branch/'world.sqlite',socket,config['budget']['tool_calls']));server.start();processes.append(server)
                faults[mode]=dict(source_fault,protocol_version='shared-fault-prefix-recovery-v1',recovery_mode=mode,inherited_fault=True,recovery_started=False,world_pid=server.pid,source_checkpoint_sha256=binding['source_checkpoint_sha256'])
                write_json(branch/'fault_controller.json',faults[mode])
            # Both world services and then both workers are started before
            # waiting for either recovery; no branch receives a new deadline.
            for mode in ('simple','full'):
                socket=Path(temporary)/(mode+'.sock')
                while not socket.exists():
                    if time.monotonic()>=deadline:raise TimeoutError('shared recovery deadline exhausted')
                    if not processes[0 if mode=='simple' else 1].is_alive():raise RuntimeError('recovery world service failed')
                    time.sleep(.01)
            for mode in ('simple','full'):
                parent,child=context.Pipe();branch=branches[mode]
                agent=context.Process(target=worker,args=(dict(config,mode=mode),task['public_task'],branch,Path(temporary)/(mode+'.sock'),child,True,origin,source))
                agent.start();child.close();processes.append(agent);channels[mode]=(parent,agent)
                faults[mode]['recovery_worker_pid']=agent.pid;write_json(branch/'fault_controller.json',faults[mode])
            while len(completed)<2 and time.monotonic()<deadline:
                for mode,(channel,agent) in channels.items():
                    if mode in completed:continue
                    if channel.poll(.01):
                        try:message=channel.recv()
                        except EOFError:
                            completed[mode]=dict(termination={'state':'INFRA_ERROR'},error={'type':'RecoveryPipeClosed'});continue
                        events[mode].append(dict(kind=message['kind'],elapsed_wall_seconds=time.monotonic()-origin))
                        if message['kind']=='recovery_started':
                            faults[mode]['recovery_started']=True;write_json(branches[mode]/'fault_controller.json',faults[mode])
                        elif message['kind']=='done':
                            completed[mode]=message;agent.join(min(2,max(0,deadline-time.monotonic())))
                            if agent.is_alive() or agent.exitcode!=0:completed[mode]=dict(termination={'state':'INFRA_ERROR'},error={'type':'RecoveryWorkerCleanupFailed'})
                        else:raise ValueError('unexpected shared recovery control event')
                    elif not agent.is_alive():completed[mode]=dict(termination={'state':'INFRA_ERROR'},error={'type':'RecoveryWorkerExited'})
            for mode in branches:
                if mode not in completed:completed[mode]=dict(termination={'state':'BUDGET_STOP'},error=None)
        except Exception as error:
            exhausted=time.monotonic()>=deadline
            for mode in branches:
                if mode not in completed:completed[mode]=dict(termination={'state':'BUDGET_STOP' if exhausted else 'INFRA_ERROR'},error=None if exhausted else {'type':type(error).__name__})
        finally:
            for channel,_ in channels.values():channel.close()
            for process in reversed(processes):
                if process.is_alive():process.kill()
                process.join(2)
            for branch in branches.values():shutil.rmtree(branch/'policy_temporary',ignore_errors=True)
    results={}
    for mode,branch in branches.items():
        if (branch/'recovery_policy_isolation.json').exists():shutil.copyfile(branch/'recovery_policy_isolation.json',branch/'policy_isolation.json')
        message=completed[mode]
        result=finalize_episode(branch,task,private,dict(config,mode=mode),versions,message['termination'],message['error'],origin,faults[mode],software_fixture,events[mode])
        result['scope']='conditional_shared_prefix_recovery_branch_not_end_to_end_trigger_comparison'
        write_json(branch/'result.json',result);results[mode]=result
    if verify_checkpoint(source)!=checkpoint:raise ValueError('shared source checkpoint changed during recovery')
    physical={field:prefix[field]+sum(max(0,r['termination'][field]-prefix[field]) for r in results.values()) for field in ('generated_tokens','input_tokens')}
    resources=dict(shared_prefix=prefix,physical_reported=physical,reserved_generated_tokens=3*config['budget']['generated_tokens'],max_generated_tokens=max_generated_tokens,usage_complete=len(results)==2 and all(r['resources']['usage_complete'] for r in results.values()),api_cost_usd=None,gpu_hours=None,note='Per-branch totals include the same source prefix; physical reported usage counts it once. Incomplete traces remain lower bounds; conservative reservations are not released.')
    return dict(pair_complete=set(results)=={'simple','full'},source_fault=source_fault,source_checkpoint_sha256=digest(source/'checkpoint.json'),branches=results,missing_branches=sorted({'simple','full'}-set(results)),resources=resources,scope='Conditional recovery comparison from one full-mode source checkpoint. Not an end-to-end simple/full injection comparison or formal fault matrix.')

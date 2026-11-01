"""Single full-mode model episode with a controller-owned commit-boundary kill.

Paired simple/full recovery and context-rebuild faults remain separate work.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from officegym.env.world import canonical
from officegym.env.rollout import clone_world
from officegym.env.service import serve,Client
from officegym.harness.loop import Harness
from officegym.harness.journal import Journal
from .artifacts import write_json
from .configured import validate_config
from .fault_checkpoint import capture_checkpoint
from .grading import grade_episode_snapshot


def policy(config):
    from officegym.harness.policy import ChatPolicy
    return ChatPolicy(config['endpoint'],config['model'],config['model_version'],max_output_tokens=config['max_output_tokens'],temperature=config.get('temperature',0),api_key_env=config.get('api_key_env'),tool_choice=config.get('tool_choice','auto'))


def worker(config,public,root,socket,channel,resume,origin,shared_checkpoint=None):
    root=Path(root);journal=isolated=None;phase='recovery' if resume else 'initial'
    try:
        budget=validate_config(config);deadline=origin+budget.wall_seconds
        def remaining():
            seconds=deadline-time.monotonic()
            if seconds<=0:raise TimeoutError('fault episode wall budget exhausted')
            return seconds
        from officegym.harness.tokenizer import PromptTokenizer
        tokenizer=PromptTokenizer.load(config['tokenizer_path'],config['tokenizer_revision'],template_kwargs=config.get('template_kwargs'))
        manifest=tokenizer.manifest()
        if resume and json.loads((root/'tokenizer_manifest.json').read_text())!=manifest:raise ValueError('recovery tokenizer changed')
        write_json(root/('recovery_tokenizer_manifest.json' if resume else 'tokenizer_manifest.json'),manifest)
        from officegym.harness.isolation import IsolatedPolicy
        # Namespace temporary chroots under this owned episode, including setup
        # failures where the controller has not yet received an attestation.
        temporary=root/'policy_temporary';temporary.mkdir(exist_ok=True);tempfile.tempdir=str(temporary)
        isolated=IsolatedPolicy(policy(config),protected_paths=[root/'private_spec.json',root/'event_rules.json',root/'initial_world.sqlite',root/'world.sqlite',Path(__file__).parents[1]/'graders/result.py'],startup_seconds=min(10,remaining()))
        write_json(root/('recovery_policy_isolation.json' if resume else 'policy_isolation.json'),isolated.evidence)
        write_json(root/(phase+'_worker_identity.json'),dict(harness_worker_pid=os.getpid(),policy_worker_pid=isolated.process.pid,isolation_version=isolated.evidence['version']))
        from officegym.harness.context import FullContext,TrainingContext,SimpleContext
        if config['mode']=='simple':
            context=SimpleContext(tokenizer,config['max_context_tokens'],config['max_output_tokens'])
        elif config.get('context_policy')=='training_full_history':
            context=TrainingContext(tokenizer,config['max_context_tokens'],config['max_output_tokens'],count_history_tokens=tokenizer.history_tokens)
        else:
            context=FullContext(tokenizer,config['max_context_tokens'],isolated,trigger_tokens=config.get('compression_trigger_tokens',8192),reserved_output_tokens=config['max_output_tokens'],archive_path=root/'summaries.jsonl')
        def call(name,args,operation=None):
            response=Client(socket,timeout=min(30,remaining())).call(name,args,operation)
            if not resume and name=='mail.send' and args.get('kind')=='invitation' and response.get('ok'):
                channel.send(dict(kind='boundary',operation_id=operation))
                channel.recv()  # Controller kills this worker before journal ACK.
            return response
        journal=Journal(root/'runtime.sqlite') if config['mode']=='full' else None;harness=Harness(isolated,call,public,root/'trajectory.jsonl',budget,journal,context)
        if resume:
            if shared_checkpoint is not None:harness.restore_fault_checkpoint(shared_checkpoint)
            else:harness.restore()
            channel.send({'kind':'recovery_started'})
            if journal is not None:harness.reconcile_pending()
        termination=harness.run() if resume else harness.run(started_at=origin)
        channel.send(dict(kind='done',termination=termination,error=None))
    except Exception as error:
        info=dict(type=type(error).__name__,phase=phase)
        write_json(root/(phase+'_worker_error.json'),info)
        exhausted=time.monotonic()>=origin+config['budget']['wall_seconds']
        channel.send(dict(kind='done',termination={'state':'BUDGET_STOP' if exhausted else 'INFRA_ERROR'},error=None if exhausted else info))
    finally:
        if journal is not None:journal.close()
        if isolated is not None:isolated.close()
        channel.close()


def controller(plan):
    root=Path(plan['episode']);root.mkdir()
    source=Path(plan['instance']);config=plan['config'];budget=validate_config(config);origin=plan['origin'];deadline=origin+budget.wall_seconds
    from .provenance import provenance
    versions=provenance(source,policy(config))
    for name in ('task.json','private_spec.json','event_rules.json','initial_world.sqlite'):
        shutil.copyfile(source/name,root/name)
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=versions['instance_artifact_sha256'][name]:raise ValueError('fault input changed during retention')
    write_json(root/'provenance.json',versions);write_json(root/'resolved_eval_config.json',config);write_json(root/'budget.json',asdict(budget))
    write_json(root/'policy_isolation.json',dict(version='policy-chroot-v1',verified=False,status='not_started'))
    task=json.loads((root/'task.json').read_text());private=json.loads((root/'private_spec.json').read_text())
    world=clone_world(root/'initial_world.sqlite',root/'world.sqlite')
    try:
        if world.db.execute('SELECT 1 FROM operations LIMIT 1').fetchone() or json.loads(world.db.execute("SELECT value_json FROM meta WHERE key='tool_calls'").fetchone()[0]) or json.loads(world.db.execute("SELECT value_json FROM meta WHERE key='finished'").fetchone()[0]):raise ValueError('fault episode requires an unexecuted initial world')
        with world.db:world.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('max_tool_calls',canonical(budget.tool_calls)))
    finally:world.close()
    context=multiprocessing.get_context('spawn');processes=[];events=[];termination={'state':'INFRA_ERROR'};error=None
    fault=dict(protocol='after_world_commit',protocol_version='fault-full-commit-v1',fault_injected=False,recovery_started=False,world_survived=False,killed_worker_exitcode=None)
    with tempfile.TemporaryDirectory(prefix='officegym-fault-socket-') as temporary:
        socket=Path(temporary)/'world.sock';server=context.Process(target=serve,args=(root/'world.sqlite',socket,budget.tool_calls));server.start();processes.append(server);fault['world_pid']=server.pid
        write_json(root/'fault_controller.json',fault)
        try:
            while not socket.exists():
                if not server.is_alive():raise RuntimeError('fault world service startup failed')
                if time.monotonic()>=deadline:raise TimeoutError('fault world startup budget exhausted')
                time.sleep(.01)
            for resume in (False,True):
                if resume and not fault['fault_injected']:break
                parent,child=context.Pipe();agent=context.Process(target=worker,args=(config,task['public_task'],root,socket,child,resume,origin));agent.start();child.close();processes.append(agent);boundary=False;done=False
                fault['recovery_worker_pid' if resume else 'initial_worker_pid']=agent.pid
                write_json(root/'fault_controller.json',fault)
                try:
                    while time.monotonic()<deadline:
                        if parent.poll(min(.05,max(0,deadline-time.monotonic()))):
                            message=parent.recv();events.append(dict(phase='recovery' if resume else 'initial',kind=message['kind'],elapsed_wall_seconds=time.monotonic()-origin))
                            if message['kind']=='recovery_started':
                                fault['recovery_started']=True;write_json(root/'fault_controller.json',fault)
                            elif message['kind']=='boundary':
                                if resume or fault['fault_injected']:raise ValueError('fault injection may occur only once')
                                agent.kill();agent.join(2)
                                if agent.is_alive() or agent.exitcode!=-9 or not server.is_alive():raise RuntimeError('fault kill/world survival not verified')
                                checkpoint=capture_checkpoint(root/'world.sqlite',root/'runtime.sqlite',root/'trajectory.jsonl',root/'committed_checkpoint')
                                if checkpoint['boundary']['operation_id']!=message['operation_id']:raise ValueError('fault controller boundary operation differs')
                                fault.update(fault_injected=True,world_survived=True,killed_worker_exitcode=agent.exitcode);write_json(root/'fault_controller.json',fault)
                                if plan.get('paired'):
                                    from .fault_pair import recover_pair
                                    pair=recover_pair(root,config,task,private,versions,origin,fault,plan.get('software_fixture',False),plan['max_generated_tokens'])
                                    write_json(root/'pair_result.json',pair)
                                    return
                                boundary=True;break
                            elif message['kind']=='done':termination=message['termination'];error=message['error'];done=True;break
                        elif not agent.is_alive():raise RuntimeError('fault worker exited without terminal result')
                    if not boundary and not done:raise TimeoutError('fault episode wall budget exhausted')
                    if done:
                        agent.join(min(2,max(0,deadline-time.monotonic())))
                        if agent.is_alive():raise TimeoutError('fault worker cleanup budget exhausted')
                        if agent.exitcode!=0:raise RuntimeError('fault worker failed after terminal message')
                        break
                finally:parent.close()
        except Exception as exc:
            exhausted=time.monotonic()>=deadline
            error=None if exhausted else dict(type=type(exc).__name__);termination={'state':'BUDGET_STOP' if exhausted else 'INFRA_ERROR'}
        finally:
            for process in reversed(processes):
                if process.is_alive():process.kill()
                process.join(2)
            shutil.rmtree(root/'policy_temporary',ignore_errors=True)
    result=finalize_episode(root,task,private,config,versions,termination,error,origin,fault,plan.get('software_fixture',False),events)
    if plan.get('paired'):
        write_json(root/'pair_result.json',dict(pair_complete=False,source_fault=fault,source_result=result,branches={},missing_branches=['simple','full'],resources=dict(result['resources'],reserved_generated_tokens=3*config['budget']['generated_tokens'],max_generated_tokens=plan['max_generated_tokens']),scope='Source episode did not reach a usable shared fault checkpoint; retain in trigger denominator.'))


def finalize_episode(root,task,private,config,versions,termination,error,origin,fault,software_fixture,events):
    budget=validate_config(config)
    # Recover usage from retained trace even after a killed or failed worker.
    trace=root/'trajectory.jsonl';records=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    prefix=records[0].get('recovery',{}).get('prior_usage',{}) if records else json.loads((root/'source_usage.json').read_text()) if (root/'source_usage.json').exists() else {}
    termination.update(generated_tokens=prefix.get('generated_tokens',0)+sum(r.get('output_tokens',0) if r['kind']=='model_response' else r.get('summary_output_tokens',0) if r['kind']=='context_selection' else 0 for r in records),input_tokens=prefix.get('input_tokens',0)+sum(r.get('input_tokens',0) if r['kind']=='model_response' else r.get('summary_input_tokens',0) if r['kind']=='context_selection' else 0 for r in records),wall_seconds=time.monotonic()-origin)
    with sqlite3.connect((root/'world.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:termination['tool_calls']=json.loads(db.execute("SELECT value_json FROM meta WHERE key='tool_calls'").fetchone()[0])
    grade,audit=grade_episode_snapshot(root/'world.sqlite',{**task['public_task'],**private},root/'initial_world.sqlite',termination_state=termination['state'],infrastructure_error=error)
    isolation=json.loads((root/'policy_isolation.json').read_text())
    result=dict(instance_id=task['instance_id'],split=task['split'],template_id=task['template_id'],mode=config['mode'],model_version=config['model_version'],policy_kind='software_fixture_fault_transport' if software_fixture else 'inference_service',termination=termination,grade=grade,operation_audit=audit,policy_isolation=isolation,infrastructure_error=error,runner_wall_seconds=time.monotonic()-origin,versions=versions,fault=fault,scope='single_full_mode_fault_episode_not_paired_or_formal_fault_matrix')
    result['resources']=dict(reserved_generated_tokens=budget.generated_tokens,reported_generated_tokens=termination['generated_tokens'],reported_input_tokens=termination['input_tokens'],usage_complete=termination['state']=='DECLARED_DONE' and error is None,api_cost_usd=None,gpu_hours=None,note='Trace totals are lower bounds after budget/infrastructure failure; full generation cap remains reserved. Null monetary/GPU fields are unmeasured.')
    write_json(root/'fault_events.json',events);write_json(root/'result.json',result)

    return result


def evaluate_fault_instance(instance,config_path,out,*,software_fixture=False):
    return _evaluate_fault(instance,config_path,out,software_fixture=software_fixture)


def evaluate_fault_pair_instance(instance,config_path,out,*,max_generated_tokens,software_fixture=False):
    return _evaluate_fault(instance,config_path,out,paired=True,max_generated_tokens=max_generated_tokens,software_fixture=software_fixture)


def _evaluate_fault(instance,config_path,out,*,paired=False,max_generated_tokens=None,software_fixture=False):
    config=json.loads(Path(config_path).read_text());budget=validate_config(config)
    if config['mode']!='full':raise ValueError('this fault runner currently supports full-mode commit recovery only')
    reserved=budget.generated_tokens
    if paired:
        validate_config(dict(config,mode='simple'))
        if type(max_generated_tokens) is not int or max_generated_tokens<3*budget.generated_tokens:
            raise ValueError('paired recovery requires an explicit global token cap covering source and both full-cap reservations')
        reserved=3*budget.generated_tokens
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    plan=dict(instance=str(Path(instance).resolve()),episode=str(root/'episode'),config=config,origin=time.monotonic(),software_fixture=software_fixture,paired=paired,max_generated_tokens=max_generated_tokens)
    write_json(root/'resource_reservations.json',dict(reserved_generated_tokens=reserved,max_generated_tokens=max_generated_tokens if paired else budget.generated_tokens,phases=['source','simple','full'] if paired else ['episode']))
    write_json(root/'controller_plan.json',plan)
    from officegym.training.process_tree import execute_tree_bounded
    with (root/'controller.stdout.log').open('w') as stdout,(root/'controller.stderr.log').open('w') as stderr:
        status=execute_tree_bounded([sys.executable,'-m','officegym.eval.fault_episode','--controller-plan',str(root/'controller_plan.json')],stdout,stderr,budget.wall_seconds+2,evidence_dir=root/'process_ownership')
    write_json(root/'execution.json',status)
    result_path=root/'episode'/('pair_result.json' if paired else 'result.json')
    if status['exit_code']!=0 or status['wall_budget_exhausted'] or not status['process_tree_cleanup'] or not result_path.is_file():
        write_json(root/'incomplete_fault_episode.json',dict(execution=status,usage_unknown=True,reserved_generated_tokens=reserved))
        raise RuntimeError('fault controller incomplete; retain owned-process and episode artifacts')
    return json.loads(result_path.read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controller-plan',required=True);args=parser.parse_args();controller(json.loads(Path(args.controller_plan).read_text()))

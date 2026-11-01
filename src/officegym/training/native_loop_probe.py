"""Actual pinned AgentLoop/tokenizer with authored token outputs, never model evidence."""
import argparse
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
from officegym.eval.artifacts import write_json
from officegym.env.world import canonical


async def run(instance,model,out):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap,ToolListWrap
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.workers.rollout.replica import TokenOutput
    from .verl_loop import OfficeGymAgentLoop
    from .verl_tool import OfficeGymTool
    from .worker_plugins import plugin_contents,verify_loop_sources
    from .registry import REGISTRY
    import inspect
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    started=time.monotonic();root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    instance=Path(instance).resolve();model=Path(model).resolve()
    before=hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()
    hashes=verify_loop_sources(Path(inspect.getfile(ToolAgentLoop)).resolve().parents[3])
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        tools=[OfficeGymTool(row['config'],OpenAIFunctionToolSchema.model_validate(row['tool_schema'])) for row in plugin_contents()['tools']['tools']]
    call='<tool_call>\n{ "name" : "board.read", "arguments" : { } }\n</tool_call><|im_end|>'
    class Server:
        def __init__(self,texts,barrier=None,cap_outputs=False,backend_steps=None):self.texts=texts;self.calls=[];self.barrier=barrier;self.cap_outputs=cap_outputs;self.backend_steps=backend_steps
        async def generate(self,**kwargs):
            # The actual server interface gets tokens, never trusted metadata.
            assert set(kwargs)<=set('request_id prompt_ids sampling_params image_data video_data mm_processor_output audio_data mm_processor_kwargs'.split())
            if not self.calls and self.barrier:
                self.barrier['arrivals']+=1
                if self.barrier['arrivals']==2:self.barrier['event'].set()
                await asyncio.wait_for(self.barrier['event'].wait(),10)
                self.barrier['paths'].update(str(e['environment']._path) for e in REGISTRY.entries.values())
            ids=tokenizer.encode(self.texts[min(len(self.calls),len(self.texts)-1)],add_special_tokens=False)
            if self.cap_outputs:ids=ids[:kwargs['sampling_params']['max_tokens']]
            assert len(ids)<=kwargs['sampling_params']['max_tokens']
            probs=[-.25-i*.001 for i in range(len(ids))]
            self.calls.append(dict(prompt_ids=list(kwargs['prompt_ids']),token_ids=ids,log_probs=probs,sampling_params=kwargs['sampling_params']))
            step=self.backend_steps[min(len(self.calls)-1,len(self.backend_steps)-1)] if self.backend_steps is not None else None
            fields=dict(global_steps=step,min_global_steps=step,max_global_steps=step) if step is not None else {}
            return TokenOutput(token_ids=ids,log_probs=probs,extra_fields=fields)
    async def case(name,texts,*,response=4096,prompt=8192,context=8192,barrier=None,reject=False,cap_outputs=False,assistant_turns=30,user_turns=30,actor_step=None,backend_steps=None,version_reject=False,parse_retries=2):
        case_root=root/name;case_root.mkdir()
        config=OmegaConf.create(dict(data={},actor_rollout_ref=dict(rollout=dict(prompt_length=prompt,response_length=response,full_determinism=False,multi_turn=dict(max_user_turns=user_turns,max_assistant_turns=assistant_turns,max_parallel_calls=1,max_tool_response_length=20000,tool_response_truncate_side='right',format='hermes')))))
        server=Server(texts,barrier,cap_outputs,backend_steps)
        loop=OfficeGymAgentLoop(trainer_config=DictConfigWrap(config),server_manager=server,tokenizer=tokenizer,processor=None,dataset_cls=object,data_config=DictConfigWrap(config.data),hf_model_type='qwen3',tools=ToolListWrap(tools))
        trusted=dict(parse_retries=parse_retries,actor_version='authored-native-protocol-fixture',max_generation_tokens=512,max_context_tokens=context,token_ledger_path=str(case_root/'tokens.sqlite'),global_rollout_token_limit=512,instance_dir=str(instance),artifact_root=str(case_root/'trajectories'))
        if actor_step is not None:
            from .actor_version import bind_actor_metadata
            trusted=bind_actor_metadata(trusted,trusted['actor_version'],actor_step)
        try:
            output=await loop.run(dict(max_tokens=128),officegym=trusted)
        except ValueError as error:
            if version_reject:
                assert 'stale or mixed actor versions' in str(error)
                write_json(case_root/'generation.json',server.calls)
                assert not REGISTRY.entries
                archive=next((case_root/'trajectories').iterdir())
                assert (archive/'incomplete.json').exists() and not (archive/'terminal.json').exists()
                tool_log=archive/'tools.jsonl'
                executed=len(tool_log.read_text().splitlines()) if tool_log.exists() else 0
                assert executed==len(server.calls)-1
                with sqlite3.connect(case_root/'tokens.sqlite') as db:
                    assert db.execute('SELECT observed FROM reservations').fetchone()==(None,)
                write_json(case_root/'rejection.json',dict(error=str(error),generation_requests=len(server.calls),reserved_usage_retained=True))
                return dict(case=name,backend_version_rejected=True,generation_requests=len(server.calls))
            if not reject:raise
            assert 'truncation forbidden' in str(error) and not server.calls
            write_json(case_root/'rejection.json',dict(error=str(error),generation_requests=0))
            return dict(case=name,rejected_before_generation=True)
        assert not reject and not version_reject
        assert output.extra_fields['officegym_actor_version_verified']==(actor_step is not None)
        write_json(case_root/'output.json',output.model_dump())
        write_json(case_root/'generation.json',server.calls)
        generated=[t for c in server.calls for t in c['token_ids']]
        actual=[t for t,m in zip(output.response_ids,output.response_mask) if m]
        assert generated==actual,'native generated IDs changed or were truncated'
        assert [p for c in server.calls for p in c['log_probs']]==[p for p,m in zip(output.response_logprobs,output.response_mask) if m]
        assert all(p==0 for p,m in zip(output.response_logprobs,output.response_mask) if not m)
        assert len(output.response_ids)==len(output.response_mask)==len(output.response_logprobs)
        assert len(output.prompt_ids)+len(output.response_ids)<=context
        terminal=json.loads(next((case_root/'trajectories').glob('*/terminal.json')).read_text())
        assert terminal['original_token_ids']==output.response_ids and terminal['response_mask']==output.response_mask
        assert terminal['original_logprobs']==output.response_logprobs
        with sqlite3.connect(case_root/'tokens.sqlite') as db:
            assert db.execute('SELECT reserved,observed FROM reservations').fetchall()==[(512,len(generated))]
        return dict(case=name,generation_requests=len(server.calls),generated_tokens=len(generated),environment_tokens=len(output.response_ids)-sum(output.response_mask),termination_reason=terminal['termination_reason'],reward=terminal['reward'],final_state_hash=terminal['final_state_hash'])
    barrier=dict(arrivals=0,event=asyncio.Event(),paths=set())
    def patch(goal):return '<tool_call>\n'+json.dumps(dict(name='board.patch',arguments=dict(expected_version=1,patch=dict(goal=goal))),ensure_ascii=False)+'\n</tool_call><|im_end|>'
    rows=list(await asyncio.gather(case('branch_a',[call,patch('探针 α'),'<tool_call>\n{"name":"task.finish","arguments":{"summary":"Authored probe stop"}}\n</tool_call><|im_end|>'],barrier=barrier),case('branch_b',[call,patch('探针 β'),'<tool_call>\n{"name":"task.finish","arguments":{"summary":"Authored probe stop"}}\n</tool_call><|im_end|>'],barrier=barrier)))
    assert len(barrier['paths'])==2 and all(not Path(p).exists() for p in barrier['paths'])
    assert rows[0]['final_state_hash']!=rows[1]['final_state_hash']
    for name,goal in [('branch_a','探针 α'),('branch_b','探针 β')]:
        dbpath=next((root/name/'trajectories').glob('*/final_world.sqlite'))
        with sqlite3.connect(dbpath) as db:
            board=db.execute('SELECT content_json FROM board').fetchone()[0]
            assert json.loads(board)['goal']==goal
    rows.append(await case('initial_prompt_overflow',[call],prompt=32,reject=True))
    rows.append(await case('initial_context_overflow',[call],context=32,reject=True))
    rows.append(await case('generated_response_limit',['long response '*200],response=40,cap_outputs=True))
    assert rows[-1]['termination_reason']=='response_limit' and rows[-1]['generated_tokens']==40
    rows.append(await case('tool_response_overflow',[call],response=40))
    assert rows[-1]['termination_reason']=='response_limit' and rows[-1]['reward']<=0
    # The observation pushes the total context over this cap, before another request.
    initial=len(json.loads((root/'branch_a/generation.json').read_text())[0]['prompt_ids'])
    first=len(tokenizer.encode(call,add_special_tokens=False))
    rows.append(await case('context_overflow',[call],context=initial+first+1))
    assert rows[-1]['termination_reason']=='generation_budget' and rows[-1]['generation_requests']==1
    rows.append(await case('parallel_calls',[call.replace('<|im_end|>','')+call]))
    assert rows[-1]['termination_reason']=='parse_error' and rows[-1]['generation_requests']==3
    rows.append(await case('assistant_turn_limit',[call],assistant_turns=1))
    assert rows[-1]['termination_reason']=='agent_turn_limit' and rows[-1]['generation_requests']==1
    rows.append(await case('actor_step_zero',[call,patch('版本零'),'<tool_call>\n{"name":"task.finish","arguments":{"summary":"Authored probe stop"}}\n</tool_call><|im_end|>'],actor_step=0,backend_steps=[0,0,0]))
    rows.append(await case('actor_step_one',[call,patch('版本一'),'<tool_call>\n{"name":"task.finish","arguments":{"summary":"Authored probe stop"}}\n</tool_call><|im_end|>'],actor_step=1,backend_steps=[1,1,1]))
    rows.append(await case('stale_actor_second_turn',[call,patch('不得执行')],actor_step=1,backend_steps=[1,0],version_reject=True))
    rows.append(await case('missing_backend_version',[call],actor_step=1,version_reject=True))
    rows.append(await case('user_turn_limit',[call],user_turns=1))
    assert rows[-1]['termination_reason']=='agent_turn_limit' and rows[-1]['generation_requests']==1
    for name,text in [('unparseable_no_call','This is not a tool call<|im_end|>'),('malformed_call','<tool_call>\n{broken JSON\n</tool_call><|im_end|>')]:
        rows.append(await case(name,[text]))
        assert rows[-1]['generation_requests']==3 and rows[-1]['termination_reason']=='parse_error'
    finish='<tool_call>\n{"name":"task.finish","arguments":{"summary":"Authored retry recovery"}}\n</tool_call><|im_end|>'
    rows.append(await case('parse_recovery',['Not a tool call<|im_end|>',call,finish]))
    assert rows[-1]['generation_requests']==3 and rows[-1]['termination_reason']=='policy_terminated'
    rows.append(await case('parse_streak_reset',['Not a tool call<|im_end|>',call,'Still invalid<|im_end|>','Still invalid<|im_end|>',finish]))
    assert rows[-1]['generation_requests']==5 and rows[-1]['termination_reason']=='policy_terminated'
    rows.append(await case('nonobject_arguments',[call.replace('{ }','[]')]))
    assert rows[-1]['generation_requests']==3 and rows[-1]['termination_reason']=='parse_error'
    rows.append(await case('parse_zero_retries',['Not a tool call<|im_end|>'],parse_retries=0))
    assert rows[-1]['generation_requests']==1 and rows[-1]['termination_reason']=='parse_error'
    rows.append(await case('parse_feedback_response_limit',['Not a tool call<|im_end|>'],response=20))
    assert rows[-1]['generation_requests']==1 and rows[-1]['termination_reason']=='response_limit'
    assert not REGISTRY.entries
    assert hashlib.sha256((instance/'initial_world.sqlite').read_bytes()).hexdigest()==before
    local_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('verl_loop.py'),Path(__file__).with_name('worker_plugins.py'),Path(__file__).with_name('actor_version.py'))}
    report=dict(passed=True,parse_retry_parity_verified=True,parse_retry_scope='Native parse exhaustion uses three generations with two bounded retries; compare ordinary Harness evidence separately. Token/mask/logprob preservation is asserted for every case.',kind='authored-token-native-loop-protocol-not-model-rollout',cases=rows,source_hashes=hashes,local_source_hashes=local_hashes,instance=str(instance),initial_snapshot_sha256=before,isolated_paths=sorted(barrier['paths']),model_generations=0,optimizer_updates=0,teacher_samples=0,wall_seconds=time.monotonic()-started)
    write_json(root/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance',required=True);parser.add_argument('--model',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        print(canonical(asyncio.run(run(args.instance,args.model,args.out))))
    else:
        import sys
        from .process_tree import execute_tree_bounded
        target=Path(args.out).resolve()
        command=[sys.executable,'-m','officegym.training.native_loop_probe','--instance',args.instance,'--model',args.model,'--out',args.out,'--worker']
        status=execute_tree_bounded(command,sys.stdout,sys.stderr,180,evidence_dir=target.with_name(target.name+'.ownership'))
        if status['exit_code'] or status['wall_budget_exhausted']:raise SystemExit(1)

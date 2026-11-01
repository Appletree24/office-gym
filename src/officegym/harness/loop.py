"""Shared policy loop. Model adapters must report actual token usage."""
import time
import os
import json
import hashlib
from pathlib import Path
from dataclasses import asdict
import uuid
from dataclasses import dataclass
from officegym.env.world import canonical
from officegym.tools.schema import WRITES, tool_schemas
from .versions import HARNESS_VERSION


def wall_clock_identity():
    # A monotonic origin is reusable only on the same boot/clock namespace.
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    namespace=Path('/proc/self/ns/time')
    identity=dict(boot=boot,time_namespace=os.readlink(namespace) if namespace.exists() else 'kernel-monotonic',implementation=time.get_clock_info('monotonic').implementation)
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


@dataclass(frozen=True)
class Budget:
    tool_calls: int = 200
    generated_tokens: int = 16000
    wall_seconds: float = 600
    parse_retries: int = 2

    def __post_init__(self):
        if type(self.tool_calls) is not int or self.tool_calls<1 or type(self.generated_tokens) is not int or self.generated_tokens<1 or type(self.parse_retries) is not int or self.parse_retries<0:
            raise ValueError('invalid finite integer budget')
        import math
        if type(self.wall_seconds) not in (int,float) or not math.isfinite(self.wall_seconds) or self.wall_seconds<=0:
            raise ValueError('invalid finite wall budget')


@dataclass
class AssistantTurn:
    tool: str
    arguments: dict
    input_tokens: int
    output_tokens: int
    raw: object = None
    content: str | None = None


def initial_messages(task,full=False):
    messages = [{'role':'system','content':'Execute the public task using tools. Return exactly one tool call per response, then wait for its observation before choosing the next call. Tool observations are evidence; do not invent confirmations.'},
                {'role':'user','content':canonical(task)}]
    if full:
        messages[0]['content'] += ' Maintain goals, constraints, evidence, blockers and waiting conditions with board tools; reconcile uncertain writes through operations.get.'
    return messages


class Harness:
    def __init__(self, policy, call, task, trace_path, budget=None, journal=None, context_manager=None):
        self.policy, self.call, self.task = policy, call, task
        if context_manager is not None and hasattr(context_manager,'policy') and context_manager.policy is not policy:
            raise ValueError('compression must use the same executing policy')
        self.context_manager = context_manager
        self.trace_path, self.budget, self.journal = trace_path, budget or Budget(), journal
        self.messages = initial_messages(task,full=journal is not None)
        self.state = 'READY'
        self.prior_usage = dict(tool_calls=0,generated_tokens=0,input_tokens=0,wall_seconds=0)
        self.restored = False
        self.pending_actions = {}
        self.parse_failures = 0
        self.wall_origin = None

    def restore(self):
        if self.restored:
            raise ValueError('context already restored')
        rows = [json.loads(line) for line in Path(self.trace_path).read_text().splitlines() if line.strip()]
        if not rows or rows[0].get('kind')!='run_header' or rows[0]['task_sha256']!=hashlib.sha256(canonical(self.task).encode()).hexdigest() or rows[0]['budget']!=asdict(self.budget):
            raise ValueError('trace task or budget does not match resumed run')
        if rows[0].get('harness_version')!=HARNESS_VERSION:
            raise ValueError('cannot resume a different harness version')
        if rows[0].get('model_version') != getattr(self.policy,'model_version',None):
            raise ValueError('cannot change model version while resuming')
        if rows[0]['mode'] != ('full' if self.journal else 'simple'):
            raise ValueError('cannot change harness mode while resuming')
        anchor=rows[0].get('wall_clock',{})
        origin=anchor.get('monotonic_origin')
        now=time.monotonic()
        import math
        if anchor.get('identity')!=wall_clock_identity() or type(origin) not in (int,float) or not math.isfinite(origin) or origin>now:
            raise ValueError('cannot resume a missing or changed monotonic clock identity')
        pending = {}
        for row in rows:
            self.prior_usage['wall_seconds'] = max(self.prior_usage['wall_seconds'],row.get('elapsed_wall_seconds',0))
            if row['kind']=='model_response':
                self.prior_usage['generated_tokens'] += row['output_tokens']
                self.prior_usage['input_tokens'] += row['input_tokens']
            elif row['kind']=='context_selection':
                self.prior_usage['generated_tokens'] += row.get('summary_output_tokens',0)
                self.prior_usage['input_tokens'] += row.get('summary_input_tokens',0)
            elif row['kind']=='parse_failure':
                self.parse_failures = row['consecutive_failures']
                if row.get('message'):
                    self.messages.append(row['message'])
            elif row['kind']=='action_start':
                self.parse_failures = 0
                pending[row['action_id']] = row
                self.prior_usage['tool_calls'] += 1
            elif row['kind']=='interaction':
                self.messages.extend(row['messages'])
                pending.pop(row.get('action_id'),None)
            elif row['kind']=='termination':
                self.prior_usage['wall_seconds'] = max(self.prior_usage['wall_seconds'],row['wall_seconds'])
                if row['state'] in ('DECLARED_DONE','PARSE_ERROR'):
                    raise ValueError('cannot resume declared-finished episode')
        now=time.monotonic()
        self.prior_usage['wall_seconds']=max(self.prior_usage['wall_seconds'],now-origin)
        # Retain any larger logged usage conservatively; never recover more time.
        self.wall_origin=min(origin,now-self.prior_usage['wall_seconds'])
        self.pending_actions = pending
        self.restored = True
        return dict(self.prior_usage)

    def reconcile_pending(self):
        if not self.restored or not self.journal:
            raise ValueError('durable full-mode restoration required')
        self.prior_usage['wall_seconds']=time.monotonic()-self.wall_origin
        with open(self.trace_path,'a') as stream:
            def append(record):
                self.prior_usage['wall_seconds'] = time.monotonic()-self.wall_origin
                record = dict(record,elapsed_wall_seconds=self.prior_usage['wall_seconds'])
                stream.write(canonical(record)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            for action_id,action in list(self.pending_actions.items()):
                if action['tool'] not in WRITES:
                    raise ValueError('uncertain read/control requires explicit episode handling')
                if self.prior_usage['tool_calls']>=self.budget.tool_calls:
                    raise ValueError('no remaining recovery tool budget')
                if time.monotonic()-self.wall_origin>=self.budget.wall_seconds:
                    raise ValueError('recovery wall budget exhausted')
                query_id = uuid.uuid4().hex
                arguments = {'operation_id':action['operation_id']}
                append({'kind':'action_start','action_id':query_id,'tool':'operations.get','arguments':arguments,'operation_id':None})
                self.prior_usage['tool_calls'] += 1
                query = self.call('operations.get',arguments,None)
                pair = [{'role':'assistant','tool_call':{'name':'operations.get','arguments':arguments,'id':query_id}}, {'role':'tool','tool_call_id':query_id,'content':canonical(query)}]
                append({'kind':'interaction','action_id':query_id,'messages':pair,'operation_id':None})
                self.messages.extend(pair)
                if not query['ok']:
                    raise ValueError('unknown operation retained after failed query')
                if query['data']['committed']:
                    result = dict(ok=True,data=query['data']['result'],error=None,meta=query['meta'])
                else:
                    if self.prior_usage['tool_calls']>=self.budget.tool_calls or time.monotonic()-self.wall_origin>=self.budget.wall_seconds:
                        raise ValueError('no remaining recovery retry budget')
                    append(dict(action,kind='action_start',retry_same_key=True))
                    self.prior_usage['tool_calls'] += 1
                    result = self.call(action['tool'],action['arguments'],action['operation_id'])
                    if result.get('error') and result['error']['code']=='IDEMPOTENCY_CONFLICT':
                        raise ValueError('recovery request hash conflict; intent retained')
                pair = [{'role':'assistant','tool_call':{'name':action['tool'],'arguments':action['arguments'],'id':action_id}}, {'role':'tool','tool_call_id':action_id,'content':canonical(result)}]
                if action.get('content') is not None:
                    pair[0]['content'] = action['content']
                append({'kind':'interaction','action_id':action_id,'messages':pair,'operation_id':action['operation_id'],'reconciled':True})
                self.messages.extend(pair)
                status = self.journal.db.execute('SELECT status FROM write_intents WHERE operation_id=?',(action['operation_id'],)).fetchone()
                if status and status[0]=='PREPARED':
                    self.journal.acknowledge(action['operation_id'],result,reconciled=True)
                del self.pending_actions[action_id]

    def run(self,*,started_at=None):
        if self.pending_actions:
            raise ValueError('unresolved actions require reconcile_pending before continuing')
        now=time.monotonic()
        if started_at is not None:
            import math
            if type(started_at) not in (int,float) or not math.isfinite(started_at) or started_at>now:
                raise ValueError('wall origin must be a finite past monotonic timestamp')
        if self.restored and started_at is not None:
            raise ValueError('recovery must retain its frozen wall clock origin')
        started=self.wall_origin if self.restored else now if started_at is None else started_at
        self.wall_origin=started
        if Path(self.trace_path).exists() and Path(self.trace_path).stat().st_size and not self.restored:
            raise ValueError('existing trace requires explicit restore')
        calls,generated,inputs = (self.prior_usage[key] for key in ('tool_calls','generated_tokens','input_tokens'))
        failures = self.parse_failures
        self.state = 'RUNNING'
        with open(self.trace_path,'a') as trace:
            def log(item):
                item = dict(item,elapsed_wall_seconds=time.monotonic()-started)
                trace.write(canonical(item)+'\n')
                trace.flush()
                os.fsync(trace.fileno())
            if not self.restored:
                log({'kind':'run_header','wall_clock':dict(monotonic_origin=started,identity=wall_clock_identity()),'harness_version':HARNESS_VERSION,'task_sha256':hashlib.sha256(canonical(self.task).encode()).hexdigest(),'budget':asdict(self.budget),'mode':'full' if self.journal else 'simple','model_version':getattr(self.policy,'model_version',None),'tool_choice':getattr(self.policy,'tool_choice',None),'initial_messages':self.messages,'tool_schemas':tool_schemas()})
            while True:
                remaining = dict(tool_calls=self.budget.tool_calls-calls, generated_tokens=self.budget.generated_tokens-generated,
                                 wall_seconds=self.budget.wall_seconds-(time.monotonic()-started))
                if any(value <= 0 for value in remaining.values()):
                    self.state = 'BUDGET_STOP'
                    break
                try:
                    model_messages = self.messages
                    if self.context_manager:
                        from .context import ContextOverflow
                        try:
                            if hasattr(self.context_manager,'set_budget'):
                                self.context_manager.set_budget(remaining)
                            model_messages,context_info = self.context_manager.prepare(self.messages,tool_schemas())
                        except ContextOverflow:
                            self.state = 'CONTEXT_OVERFLOW'
                            break
                        log({'kind':'context_selection',**context_info})
                        generated += context_info.get('summary_output_tokens',0)
                        inputs += context_info.get('summary_input_tokens',0)
                        remaining['generated_tokens'] = self.budget.generated_tokens-generated
                        remaining['wall_seconds'] = self.budget.wall_seconds-(time.monotonic()-started)
                        if remaining['generated_tokens']<=0 or remaining['wall_seconds']<=0:
                            self.state = 'BUDGET_STOP'
                            break
                    turn = self.policy.generate(model_messages,tool_schemas(),remaining)
                except Exception as exc:
                    exhausted=time.monotonic()-started>=self.budget.wall_seconds
                    log({'kind':'wall_budget_exhausted' if exhausted else 'infrastructure_error','error_type':type(exc).__name__})
                    self.state = 'BUDGET_STOP' if exhausted else 'INFRA_ERROR'
                    break
                if not isinstance(turn,AssistantTurn) or type(turn.input_tokens) is not int or type(turn.output_tokens) is not int or min(turn.input_tokens,turn.output_tokens)<0:
                    raise ValueError('policy must return AssistantTurn with measured token usage')
                generated += turn.output_tokens
                inputs += turn.input_tokens
                log({'kind':'model_response','tool':turn.tool,'arguments':turn.arguments,'raw':turn.raw,'content':turn.content,'input_tokens':turn.input_tokens,'output_tokens':turn.output_tokens})
                if generated > self.budget.generated_tokens or time.monotonic()-started >= self.budget.wall_seconds:
                    self.state = 'BUDGET_STOP'
                    break
                if not isinstance(turn.tool,str) or not isinstance(turn.arguments,dict):
                    failures += 1
                    message = {'role':'user','content':'Return one valid tool call with object arguments.'}
                    log({'kind':'parse_failure','consecutive_failures':failures,'message':message if failures<=self.budget.parse_retries else None})
                    if failures > self.budget.parse_retries:
                        self.state = 'PARSE_ERROR'
                        break
                    self.messages.append(message)
                    continue
                failures = 0
                operation = self.journal.prepare(turn.tool,turn.arguments) if self.journal and turn.tool in WRITES else uuid.uuid4().hex if turn.tool in WRITES else None
                action_id = uuid.uuid4().hex
                log({'kind':'action_start','action_id':action_id,'tool':turn.tool,'arguments':turn.arguments,'operation_id':operation,'content':turn.content})
                calls += 1
                try:
                    result = self.call(turn.tool,turn.arguments,operation)
                except Exception as exc:
                    exhausted=time.monotonic()-started>=self.budget.wall_seconds
                    log({'kind':'wall_budget_exhausted' if exhausted else 'infrastructure_error','error_type':type(exc).__name__,'operation_id':operation})
                    self.state = 'BUDGET_STOP' if exhausted else 'INFRA_ERROR'
                    break
                if self.journal and turn.tool in WRITES:
                    self.journal.acknowledge(operation,result)
                pair = [{'role':'assistant','tool_call':{'name':turn.tool,'arguments':turn.arguments,'id':str(calls)}},
                        {'role':'tool','tool_call_id':str(calls),'content':canonical(result)}]
                if turn.content is not None:
                    pair[0]['content'] = turn.content
                self.messages.extend(pair)
                log({'kind':'interaction','messages':pair,'operation_id':operation,'action_id':action_id})
                if time.monotonic()-started>=self.budget.wall_seconds:
                    self.state='BUDGET_STOP'
                    break
                if self.context_manager is not None and hasattr(self.context_manager,'validate_history'):
                    from .context import ContextOverflow
                    try:
                        history_info=self.context_manager.validate_history(self.messages,tool_schemas())
                    except ContextOverflow as error:
                        log({'kind':'context_history_overflow','context_policy':'training_full_history','full_history_tokens':getattr(error,'full_history_tokens',None),'max_history_tokens':getattr(error,'max_history_tokens',None)})
                        self.state='CONTEXT_OVERFLOW'
                        break
                    log({'kind':'context_history_validation',**history_info})
                if turn.tool == 'task.finish' and result['ok']:
                    self.state = 'DECLARED_DONE'
                    break
                if result.get('error') and result['error']['code']=='BUDGET_EXHAUSTED':
                    self.state = 'BUDGET_STOP'
                    break
            summary = dict(state=self.state,tool_calls=calls,generated_tokens=generated,input_tokens=inputs,wall_seconds=time.monotonic()-started)
            log({'kind':'termination',**summary})
            return summary

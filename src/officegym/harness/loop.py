"""Shared policy loop. Model adapters must report actual token usage."""
import time
import uuid
from dataclasses import dataclass
from officegym.env.world import canonical
from officegym.tools.schema import WRITES, tool_schemas


@dataclass(frozen=True)
class Budget:
    tool_calls: int = 200
    generated_tokens: int = 16000
    wall_seconds: float = 600
    parse_retries: int = 2


@dataclass
class AssistantTurn:
    tool: str
    arguments: dict
    input_tokens: int
    output_tokens: int
    raw: object = None


class Harness:
    def __init__(self, policy, call, task, trace_path, budget=None, journal=None, context_manager=None):
        self.policy, self.call, self.task = policy, call, task
        self.context_manager = context_manager
        self.trace_path, self.budget, self.journal = trace_path, budget or Budget(), journal
        self.messages = [{'role':'system','content':'Execute the public task using tools. Tool observations are evidence; do not invent confirmations.'},
                         {'role':'user','content':canonical(task)}]
        if journal:
            self.messages[0]['content'] += ' Maintain goals, constraints, evidence, blockers and waiting conditions with board tools; reconcile uncertain writes through operations.get.'
        self.state = 'READY'

    def run(self):
        started = time.monotonic()
        calls = generated = inputs = failures = 0
        self.state = 'RUNNING'
        with open(self.trace_path,'a') as trace:
            def log(item):
                trace.write(canonical(item)+'\n')
                trace.flush()
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
                            model_messages,context_info = self.context_manager.prepare(self.messages,tool_schemas())
                        except ContextOverflow:
                            self.state = 'CONTEXT_OVERFLOW'
                            break
                        log({'kind':'context_selection',**context_info})
                    turn = self.policy.generate(model_messages,tool_schemas(),remaining)
                except Exception as exc:
                    log({'kind':'infrastructure_error','error_type':type(exc).__name__})
                    self.state = 'INFRA_ERROR'
                    break
                if not isinstance(turn,AssistantTurn) or type(turn.input_tokens) is not int or type(turn.output_tokens) is not int or min(turn.input_tokens,turn.output_tokens)<0:
                    raise ValueError('policy must return AssistantTurn with measured token usage')
                generated += turn.output_tokens
                inputs += turn.input_tokens
                log({'kind':'model_response','tool':turn.tool,'arguments':turn.arguments,'raw':turn.raw,'input_tokens':turn.input_tokens,'output_tokens':turn.output_tokens})
                if generated > self.budget.generated_tokens or time.monotonic()-started >= self.budget.wall_seconds:
                    self.state = 'BUDGET_STOP'
                    break
                if not isinstance(turn.tool,str) or not isinstance(turn.arguments,dict):
                    failures += 1
                    if failures > self.budget.parse_retries:
                        self.state = 'PARSE_ERROR'
                        break
                    self.messages.append({'role':'user','content':'Return one valid tool call with object arguments.'})
                    continue
                failures = 0
                operation = self.journal.prepare(turn.tool,turn.arguments) if self.journal and turn.tool in WRITES else uuid.uuid4().hex if turn.tool in WRITES else None
                try:
                    result = self.call(turn.tool,turn.arguments,operation)
                except Exception as exc:
                    log({'kind':'infrastructure_error','error_type':type(exc).__name__,'operation_id':operation})
                    self.state = 'INFRA_ERROR'
                    break
                calls += 1
                if self.journal and turn.tool in WRITES:
                    self.journal.acknowledge(operation,result)
                pair = [{'role':'assistant','tool_call':{'name':turn.tool,'arguments':turn.arguments,'id':str(calls)}},
                        {'role':'tool','tool_call_id':str(calls),'content':canonical(result)}]
                self.messages.extend(pair)
                log({'kind':'interaction','messages':pair,'operation_id':operation})
                if turn.tool == 'task.finish' and result['ok']:
                    self.state = 'DECLARED_DONE'
                    break
                if result.get('error') and result['error']['code']=='BUDGET_EXHAUSTED':
                    self.state = 'BUDGET_STOP'
                    break
            summary = dict(state=self.state,tool_calls=calls,generated_tokens=generated,input_tokens=inputs,wall_seconds=time.monotonic()-started)
            log({'kind':'termination',**summary})
            return summary

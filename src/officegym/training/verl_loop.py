"""Fixed-source AgentLoop integration, pending real verl/vLLM smoke tests.

Inherits the pinned upstream Continuous Token path: generated IDs are not
reconstructed from decoded tool calls. The training launcher must enable logprobs.
"""
from uuid import uuid4
from pathlib import Path
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop,AgentState
from officegym.harness.loop import initial_messages
from officegym.tools.schema import tool_schemas
from .registry import REGISTRY
from .token_budget import TokenLedger,GenerationBudget


@register('officegym')
class OfficeGymAgentLoop(ToolAgentLoop):
    async def run(self,sampling_params,priority=0,**kwargs):
        trusted = kwargs['officegym']
        actor = trusted['actor_version']
        trajectory = uuid4().hex
        self._officegym_trajectory = trajectory
        self._generation_budget = GenerationBudget(trusted['max_generation_tokens'],trusted['max_context_tokens'])
        self._generation_uncertain = False
        ledger = TokenLedger(trusted['token_ledger_path'],trusted['global_rollout_token_limit'])
        try:
            ledger.reserve(trajectory,actor,trusted['max_generation_tokens'])
            REGISTRY.acquire(trajectory,'__episode__',trusted['instance_dir'],actor,trusted.get('max_tool_calls',24),artifact_dir=Path(trusted['artifact_root'])/trajectory)
            environment = REGISTRY.entries[trajectory]['environment']
            expected_tools = {tool['name'] for tool in tool_schemas()}
            if set(self.tools)!=expected_tools:
                raise ValueError('training must expose exactly the same 24 OfficeGym tools')
            create = dict(trajectory_id=trajectory,officegym_instance_dir=trusted['instance_dir'],actor_version=actor,max_tool_calls=trusted.get('max_tool_calls',24))
            kwargs['tools_kwargs'] = {name:{'create_kwargs':create} for name in expected_tools}
            kwargs['raw_prompt'] = initial_messages(environment.public_task,full=True)
            output = await super().run(sampling_params,priority=priority,**kwargs)
            if output.response_logprobs is None:
                raise ValueError('native rollout logprobs required; enable backend reporting')
            terminal = REGISTRY.finalize(trajectory,input_ids=output.response_ids,response_mask=output.response_mask,
                                         token_roles=['assistant' if bit else 'tool' for bit in output.response_mask],
                                         logprobs=output.response_logprobs,actor_version=actor)
            reward = terminal['reward']
            # Upstream response cap can truncate output. Such runs cannot count as success.
            if len(output.response_ids)>=self.response_length or self._generation_budget.stopped:
                reward = min(reward,0.0)
            output.reward_score = reward
            output.extra_fields.update(officegym_actor_version=actor,officegym_final_hash=terminal['final_state_hash'],officegym_tool_calls=terminal['tool_calls'],officegym_generated_tokens=self._generation_budget.observed)
            return output
        finally:
            if not self._generation_uncertain:
                row = ledger.db.execute('SELECT 1 FROM reservations WHERE trajectory=?',(trajectory,)).fetchone()
                if row:
                    ledger.record(trajectory,self._generation_budget.observed)
            ledger.close()
            REGISTRY.release(trajectory,'__episode__')

    async def _handle_generating_state(self,agent_data,sampling_params,ignore_termination=False):
        requested = sampling_params.get('max_tokens',self.response_length)
        cap = self._generation_budget.next_cap(len(agent_data.prompt_ids),requested)
        if cap==0:
            return AgentState.TERMINATED
        self._generation_uncertain = True
        state = await super()._handle_generating_state(agent_data,{**sampling_params,'max_tokens':cap},ignore_termination=ignore_termination)
        self._generation_budget.record(len(agent_data.response_ids),cap)
        self._generation_uncertain = False
        return state

    async def _handle_processing_tools_state(self,agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        environment = REGISTRY.entries[self._officegym_trajectory]['environment']
        if environment._client.meta('finished'):
            return AgentState.TERMINATED
        return state

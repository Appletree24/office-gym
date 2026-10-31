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
from .tool_execution import dispatch_generated_call,single_call
from .worker_plugins import plugin_contents,verify_loop_sources
from verl.tools.schemas import ToolResponse
from officegym.env.world import canonical


@register('officegym')
class OfficeGymAgentLoop(ToolAgentLoop):
    def __init__(self,*args,**kwargs):
        import inspect
        verify_loop_sources(Path(inspect.getfile(ToolAgentLoop)).resolve().parents[3])
        super().__init__(*args,**kwargs)

    async def run(self,sampling_params,priority=0,**kwargs):
        trusted = kwargs['officegym']
        actor = trusted['actor_version']
        from .actor_version import validate_actor_metadata,VersionedServer
        initial=self.config.get('trainer',{}).get('officegym_starting_actor_hash')
        checked_step=validate_actor_metadata(trusted,initial) if initial is not None or 'actor_update_step' in trusted else None
        original_server=self.server_manager
        version_guard=None
        trajectory = uuid4().hex
        self._officegym_trajectory = trajectory
        self._generation_budget = GenerationBudget(trusted['max_generation_tokens'],trusted['max_context_tokens'])
        self._generation_uncertain = False
        self._invalid_tool_batch = False
        self._context_response_limit = False
        self._turn_limit_reached = False
        ledger = TokenLedger(trusted['token_ledger_path'],trusted['global_rollout_token_limit'])
        try:
            ledger.reserve(trajectory,actor,trusted['max_generation_tokens'])
            REGISTRY.acquire(trajectory,'__episode__',trusted['instance_dir'],actor,trusted.get('max_tool_calls',24),artifact_dir=Path(trusted['artifact_root'])/trajectory)
            environment = REGISTRY.entries[trajectory]['environment']
            if checked_step is not None:
                version_guard=VersionedServer(original_server,checked_step,Path(trusted['artifact_root'])/trajectory)
                self.server_manager=version_guard
            expected_tools = {tool['name'] for tool in tool_schemas()}
            if set(self.tools)!=expected_tools:
                raise ValueError('training must expose exactly the same 24 OfficeGym tools')
            if (kwargs.get('extra_info') or {}).get('tool_selection'):
                raise ValueError('per-sample tool filtering is not permitted')
            # Upstream Pydantic tool models discard items/additionalProperties.
            # Feed the canonical schemas directly to the continuous-token prompt.
            self.tool_schemas = [row['tool_schema'] for row in plugin_contents()['tools']['tools']]
            create = dict(trajectory_id=trajectory,officegym_instance_dir=trusted['instance_dir'],actor_version=actor,max_tool_calls=trusted.get('max_tool_calls',24))
            kwargs['tools_kwargs'] = {name:{'create_kwargs':create} for name in expected_tools}
            kwargs['raw_prompt'] = initial_messages(environment.public_task,full=True)
            output = await super().run(sampling_params,priority=priority,**kwargs)
            if output.response_logprobs is None:
                raise ValueError('native rollout logprobs required; enable backend reporting')
            reason = ('invalid_tool_batch' if self._invalid_tool_batch else 'generation_budget' if self._generation_budget.stopped else 'response_limit' if self._context_response_limit or len(output.response_ids)>=self.response_length else 'agent_turn_limit' if self._turn_limit_reached else 'policy_terminated')
            terminal = REGISTRY.finalize(trajectory,input_ids=output.response_ids,response_mask=output.response_mask,
                                         token_roles=['assistant' if bit else 'tool' for bit in output.response_mask],
                                         logprobs=output.response_logprobs,actor_version=actor,termination_reason=reason)
            output.reward_score = terminal['reward']
            output.extra_fields.update(officegym_termination_reason=reason,officegym_actor_version=actor,officegym_final_hash=terminal['final_state_hash'],officegym_tool_calls=terminal['tool_calls'],officegym_generated_tokens=self._generation_budget.observed,officegym_invalid_tool_batch=self._invalid_tool_batch)
            output.extra_fields.update(officegym_actor_update_step=checked_step,officegym_actor_version_verified=version_guard is not None and version_guard.checks>0,officegym_trajectory_id=trajectory)
            return output
        finally:
            self.server_manager=original_server
            try:
                if not self._generation_uncertain:
                    row = ledger.db.execute('SELECT 1 FROM reservations WHERE trajectory=?',(trajectory,)).fetchone()
                    if row:
                        ledger.record(trajectory,self._generation_budget.observed)
            finally:
                try:
                    ledger.close()
                finally:
                    REGISTRY.release(trajectory,'__episode__')

    def _cap_text_prompt_length(self,prompt_ids):
        # Upstream silently left-truncates text prompts. P0 requires the whole
        # public task/schema/history, so reject this configuration before serving.
        if len(prompt_ids)>self.rollout_config.prompt_length:
            raise ValueError('OfficeGym initial prompt exceeds native prompt capacity; truncation forbidden')
        if len(prompt_ids)>=self._generation_budget.context_limit:
            raise ValueError('OfficeGym initial prompt leaves no generation context; truncation forbidden')
        return prompt_ids

    async def ct_merge_context_msg(self,*args,**kwargs):
        result=await super().ct_merge_context_msg(*args,**kwargs)
        merge,mask,_=result
        # Upstream returns TERMINATED before committing an over-cap observation
        # to AgentData. Its old output length alone cannot reveal that overflow.
        if len(mask)>=self.response_length:
            self._context_response_limit=True
        if len(merge.token_ids)>self._generation_budget.context_limit:
            self._generation_budget.stopped=True
        return result

    async def _handle_generating_state(self,agent_data,sampling_params,ignore_termination=False):
        if (self.max_assistant_turns and agent_data.assistant_turns>=self.max_assistant_turns) or (self.max_user_turns and agent_data.user_turns>=self.max_user_turns):
            self._turn_limit_reached=True
            return AgentState.TERMINATED
        requested = sampling_params.get('max_tokens',self.response_length)
        remaining=self.response_length-len(agent_data.response_mask)
        if remaining<=0:
            self._context_response_limit=True
            return AgentState.TERMINATED
        requested=min(requested,remaining)
        cap = self._generation_budget.next_cap(len(agent_data.prompt_ids),requested)
        if cap==0:
            return AgentState.TERMINATED
        self._generation_uncertain = True
        state = await super()._handle_generating_state(agent_data,{**sampling_params,'max_tokens':cap},ignore_termination=ignore_termination)
        self._generation_budget.record(len(agent_data.response_ids),cap)
        self._generation_uncertain = False
        if state==AgentState.TERMINATED and ((self.max_assistant_turns and agent_data.assistant_turns>=self.max_assistant_turns) or (self.max_user_turns and agent_data.user_turns>=self.max_user_turns)):
            self._turn_limit_reached=True
        return state

    async def _call_tool(self,tool_call,tools_kwargs,agent_data):
        # The episode lease owns the world. Avoid upstream parse/name errors that
        # bypass billing, and its character truncation of structured observations.
        environment = REGISTRY.entries[self._officegym_trajectory]['environment']
        result = dispatch_generated_call(environment,tool_call.name,tool_call.arguments)
        return ToolResponse(text=canonical(result)),0.0,{'officegym_tool_ok':float(result['ok'])}

    async def _handle_processing_tools_state(self,agent_data):
        if not single_call(agent_data.tool_calls):
            self._invalid_tool_batch = True
            return AgentState.TERMINATED
        before=(list(agent_data.prompt_ids),list(agent_data.response_mask),list(agent_data.response_logprobs))
        state = await super()._handle_processing_tools_state(agent_data)
        if self._generation_budget.stopped:
            # The oversized observation was never sent to generation. Keep the
            # exact previously observed policy trajectory; archive the executed
            # action/result and final snapshot separately and score budget failure.
            agent_data.prompt_ids,agent_data.response_mask,agent_data.response_logprobs=before
            return AgentState.TERMINATED
        environment = REGISTRY.entries[self._officegym_trajectory]['environment']
        if environment._client.meta('finished'):
            return AgentState.TERMINATED
        return state

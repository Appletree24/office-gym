"""BaseTool adapter for inspected verl commit 8df5baeed0f5ecdd7f870d83bc3d79194816e2a9.

Not runtime-smoke-verified. Custom trusted AgentLoop must retain a registry __episode__ lease for the complete rollout, pass one shared
trajectory_id to all 24 create calls, and original token data to calc_reward.
"""
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse
from officegym.env.world import canonical
from .registry import REGISTRY


class OfficeGymTool(BaseTool):
    async def create(self,instance_id=None,**kwargs):
        kwargs = kwargs.get('create_kwargs',kwargs)
        trajectory_id = kwargs['trajectory_id']
        if instance_id is not None and instance_id!=trajectory_id:
            raise ValueError('all OfficeGym tools must share the trajectory identity')
        REGISTRY.acquire(trajectory_id,self.name,kwargs['officegym_instance_dir'],kwargs['actor_version'],kwargs.get('max_tool_calls',24))
        return trajectory_id,ToolResponse()

    async def execute(self,instance_id,parameters,**kwargs):
        result = REGISTRY.call(instance_id,self.name,parameters)
        return ToolResponse(text=canonical(result)),0.0,{'officegym_tool_ok':float(result['ok'])}

    async def calc_reward(self,instance_id,**kwargs):
        # Only one tool contributes terminal reward; all intermediate rewards zero.
        if self.name!='task.finish':
            return 0.0
        terminal = REGISTRY.finalize(instance_id,**kwargs['officegym_terminal_tokens'])
        return float(terminal['reward'])

    async def release(self,instance_id,**kwargs):
        REGISTRY.release(instance_id,self.name)

"""Worker-local shared rollout state for all tool adapters in one trajectory."""
from pathlib import Path
from .environment import RolloutEnvironment


class RolloutRegistry:
    def __init__(self):
        self.entries = {}

    def acquire(self,trajectory_id,tool_name,instance_dir,actor_version,max_tool_calls=24,artifact_dir=None):
        if not trajectory_id:
            raise ValueError('trusted trajectory identity required')
        identity = (str(Path(instance_dir).resolve()),actor_version,max_tool_calls)
        if trajectory_id not in self.entries:
            environment = RolloutEnvironment(instance_dir,actor_version=actor_version,max_tool_calls=max_tool_calls,artifact_dir=artifact_dir)
            self.entries[trajectory_id] = {'identity':identity,'environment':environment,'tools':set(),'terminal':None}
        entry = self.entries[trajectory_id]
        if entry['identity']!=identity:
            raise ValueError('trajectory cannot change initial task, actor or budget')
        entry['tools'].add(tool_name)
        return trajectory_id

    def call(self,trajectory_id,name,arguments):
        entry = self.entries[trajectory_id]
        if name not in entry['tools']:
            raise ValueError('tool lifecycle not initialized')
        return entry['environment'].call(name,arguments)

    def finalize(self,trajectory_id,**token_data):
        entry = self.entries[trajectory_id]
        if entry['terminal'] is None:
            entry['terminal'] = entry['environment'].finalize(**token_data)
        return entry['terminal']

    def release(self,trajectory_id,tool_name):
        entry = self.entries.get(trajectory_id)
        if entry is None:
            return
        entry['tools'].discard(tool_name)
        if not entry['tools']:
            entry['environment'].close()
            del self.entries[trajectory_id]


REGISTRY = RolloutRegistry()

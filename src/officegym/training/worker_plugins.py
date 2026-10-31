"""Pinned verl worker registration files, without importing the GPU stack."""
import hashlib
import json
from pathlib import Path
from officegym.env.world import canonical
from officegym.tools.schema import tool_schemas


def verify_loop_sources(checkout):
    root=Path(checkout).resolve()
    hashes=json.loads(Path(__file__).with_name('verl_loop_sources.json').read_text())
    for name,expected in hashes.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('pinned native AgentLoop source differs: '+name)
    return hashes


def plugin_contents():
    tools=[]
    for tool in tool_schemas():
        tools.append(dict(class_name='officegym.training.verl_tool.OfficeGymTool',config={'type':'native'},
                          tool_schema={'type':'function','function':{'name':tool['name'],'description':tool['description'],'parameters':tool['inputSchema']}}))
    return dict(agent_loop=[{'name':'officegym','_target_':'officegym.training.verl_loop.OfficeGymAgentLoop'}],tools={'tools':tools})


def write_plugins(root):
    root=Path(root).resolve()
    records={}
    for name,contents in plugin_contents().items():
        path=root/(name+'.json')
        with path.open('x') as stream:
            stream.write(canonical(contents)+'\n')
        records[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return records


def validate_plugins(records,values):
    fields={'agent_loop':'actor_rollout_ref.rollout.agent.agent_loop_config_path',
            'tools':'actor_rollout_ref.rollout.multi_turn.tool_config_path'}
    for name,contents in plugin_contents().items():
        record=records.get(name,{})
        path=Path(record.get('path',''))
        if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record.get('sha256'):
            raise ValueError('missing or changed worker plugin file: '+name)
        if json.loads(path.read_text())!=contents:
            raise ValueError('worker plugin differs from OfficeGym contract: '+name)
        if values.get(fields[name])!=str(path):
            raise ValueError('resolved config does not load bound worker plugin: '+name)
    if values.get('actor_rollout_ref.rollout.multi_turn.function_tool_path'):
        raise ValueError('extra function tools are not allowed in OfficeGym rollout')

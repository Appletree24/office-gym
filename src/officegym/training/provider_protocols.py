"""Lossless recorded JSON-action transport for providers with parallel tool output."""
import copy
import json


def encode_single_action(payload):
    value=copy.deepcopy(payload);tools=value.pop('tools',[])
    names={tool['function']['name'] for tool in tools}
    if not tools:return value,names
    instruction='\nRespond with exactly one JSON object with exactly these keys: "name" (one tool name) and "arguments" (one object). Never return an array, a batch, multiple actions, or an ordinary answer. Use task.finish when done. The following are the available tools:\n'+json.dumps([tool['function'] for tool in tools],ensure_ascii=False,separators=(',',':'))
    messages=[]
    for message in value['messages']:
        if message.get('tool_calls'):
            calls=message['tool_calls']
            if len(calls)!=1:raise ValueError('JSON-action history requires one original tool call per assistant turn')
            function=calls[0]['function'];action=dict(name=function['name'],arguments=json.loads(function['arguments']))
            content=json.dumps(action,ensure_ascii=False,separators=(',',':'))
            if message.get('content'):content=message['content']+'\n'+content
            messages.append(dict(role='assistant',content=content))
        elif message['role']=='tool':messages.append(dict(role='user',content='Actual tool observation ('+message['tool_call_id']+'):\n'+message['content']))
        else:messages.append(copy.deepcopy(message))
    if messages[0]['role']=='system':messages[0]['content']+=instruction
    else:messages.insert(0,dict(role='system',content=instruction.lstrip()))
    value['messages']=messages;value.pop('tool_choice',None);value.pop('parallel_tool_calls',None);value['response_format']={'type':'json_object'}
    return value,names


def decode_single_action(response,names):
    value=copy.deepcopy(response)
    for index,choice in enumerate(value.get('choices',[])):
        message=choice.get('message',{})
        try:action=json.loads(message.get('content',''))
        except (ValueError,TypeError):continue
        if not isinstance(action,dict) or set(action)!={'name','arguments'} or not isinstance(action['name'],str) or action['name'] not in names or not isinstance(action['arguments'],dict):continue
        # The original content is retained separately by the gateway; this is
        # protocol normalization, not an additional assistant text prediction.
        message['content']=None
        message['tool_calls']=[dict(id='json_action_'+str(index),type='function',function=dict(name=action['name'],arguments=json.dumps(action['arguments'],ensure_ascii=False)))]
    return value

"""Chat-completions protocol adapter for a configured inference service.

No automatic retries or hidden model substitution. Server-reported usage is
mandatory. This adapter is evaluation/SFT collection only, not an RL token backend.
"""
import json
import os
from urllib.request import Request,urlopen
from officegym.env.world import canonical
from .loop import AssistantTurn


def render_messages(messages):
    rendered = []
    for message in messages:
        if 'tool_call' in message:
            call = message['tool_call']
            rendered.append({'role':'assistant','content':None,'tool_calls':[{'id':call['id'],'type':'function','function':{'name':call['name'],'arguments':canonical(call['arguments'])}}]})
        elif message['role']=='tool':
            rendered.append({'role':'tool','tool_call_id':message['tool_call_id'],'content':message['content']})
        else:
            rendered.append({'role':message['role'],'content':message.get('content','')})
    return rendered


class ChatPolicy:
    policy_kind = 'inference_service'
    def __init__(self,endpoint,model,model_version,*,api_key_env=None,max_output_tokens=1024,temperature=0,transport=None):
        if not model or not model_version or type(max_output_tokens) is not int or max_output_tokens<1:
            raise ValueError('locked model identity and finite output cap required')
        self.endpoint,self.model,self.model_version = endpoint,model,model_version
        self.api_key_env,self.max_output_tokens,self.temperature = api_key_env,max_output_tokens,temperature
        self.transport = transport or self._post

    def _post(self,payload,timeout):
        headers = {'Content-Type':'application/json'}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError('configured credential environment variable is unset')
            headers['Authorization'] = 'Bearer '+key
        request = Request(self.endpoint,data=canonical(payload).encode(),headers=headers,method='POST')
        with urlopen(request,timeout=timeout) as response:
            return json.load(response)

    def _request(self,messages,remaining,tools=None):
        limit = min(self.max_output_tokens,remaining['generated_tokens'])
        if limit<1 or remaining['wall_seconds']<=0:
            raise ValueError('generation budget exhausted')
        payload = {'model':self.model,'messages':messages,'temperature':self.temperature,'max_tokens':limit,'stream':False}
        if tools is not None:
            payload.update(tools=[{'type':'function','function':{'name':tool['name'],'description':tool['description'],'parameters':tool['inputSchema']}} for tool in tools],tool_choice='required',parallel_tool_calls=False)
        response = self.transport(payload,remaining['wall_seconds'])
        usage = response.get('usage',{})
        if any(type(usage.get(key)) is not int or usage[key]<0 for key in ('prompt_tokens','completion_tokens')):
            raise ValueError('missing measured model token usage')
        return response,usage

    def generate(self,messages,tool_schemas,remaining_budget):
        response,usage = self._request(render_messages(messages),remaining_budget,tool_schemas)
        calls = response['choices'][0]['message'].get('tool_calls',[])
        name,args = None,None
        if len(calls)==1:
            function = calls[0].get('function',{})
            name = function.get('name')
            try:
                args = json.loads(function.get('arguments',''))
            except (ValueError,TypeError):
                pass
        return AssistantTurn(name,args,usage['prompt_tokens'],usage['completion_tokens'],raw=response)

    def summarize(self,messages,remaining_budget):
        from .context import FullContext
        prompt = [{'role':'system','content':'Summarize prior interactions as a JSON object with list fields '+','.join(sorted(FullContext.REQUIRED))+'. Evidence entries need source_refs. Separate confirmed facts from hypotheses. Preserve uncertain operations. Never infer approval from a request.'}, {'role':'user','content':canonical(messages)}]
        response,usage = self._request(prompt,remaining_budget)
        try:
            summary = json.loads(response['choices'][0]['message']['content'])
        except (ValueError,TypeError):
            summary = {}
        return dict(summary=summary,input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],model_version=self.model_version,sampling_parameters={'temperature':self.temperature})

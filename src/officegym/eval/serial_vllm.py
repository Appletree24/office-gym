"""Version-locked vLLM launcher honoring parallel_tool_calls=False.

Upstream 0.11's required-tool grammar omits maxItems. Constrain that grammar
before sampling; never discard, reorder or execute extra generated calls.
"""
import copy
import importlib.metadata
import runpy

ADAPTER_VERSION='vllm-0.11-required-single-call-v1'


def serial_schema(schema,tool_choice,parallel_tool_calls):
    if tool_choice!='required' or parallel_tool_calls is not False:
        return schema
    if not isinstance(schema,dict) or schema.get('type')!='array' or schema.get('minItems')!=1:
        raise ValueError('unexpected upstream required-tool grammar')
    result=copy.deepcopy(schema)
    result['maxItems']=1
    return result


def main():
    if importlib.metadata.version('vllm')!='0.11.0':
        raise ValueError('single-call adapter requires locked vLLM 0.11.0')
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    original=ChatCompletionRequest._get_json_schema_from_tool
    def get_schema(request):
        return serial_schema(original(request),request.tool_choice,request.parallel_tool_calls)
    ChatCompletionRequest._get_json_schema_from_tool=get_schema
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')


if __name__=='__main__':main()

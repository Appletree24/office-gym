"""Shared public input schemas and strict validation for all transport adapters."""
import json
from officegym.resources import read_text

CONTRACT = json.loads(read_text('tools.json'))
TOOL_SCHEMA_VERSION=CONTRACT['version']
STRINGS = {'query','thread_id','meeting_id','document_id','approval_id','operation_id','kind','subject','body','title','content','reason','assignee_id','source_meeting_id','todo_id','approver_id','summary','cursor','after_message_id'}
INTEGERS = {'start_at','end_at','version','expected_version','meeting_version','due_at','until','after_seq','limit'}
ARRAYS = {'person_ids','participant_ids','recipient_ids','document_refs','attachment_refs'}
OPTIONAL = {
    'people.search':['limit','cursor'], 'calendar.query':['limit','cursor'],
    'mail.search':['limit','cursor'], 'mail.get_thread':['limit','after_message_id'],
    'documents.search':['limit','cursor'], 'documents.read':['limit','cursor'],
    'todos.search':['limit','cursor'], 'events.poll':['limit'],
    'calendar.create':['document_refs'],
    'mail.send':['meeting_id','meeting_version','approval_id','thread_id','attachment_refs','request'],
}


def tool_schemas():
    result = []
    for tool in CONTRACT['tools']:
        fields = tool['required_arguments'] + OPTIONAL.get(tool['name'],[])
        properties = {}
        for key in fields:
            kind = 'string' if key in STRINGS else 'integer' if key in INTEGERS else 'array' if key in ARRAYS else 'object'
            properties[key] = {'type':kind}
            if key in ('person_ids','participant_ids','recipient_ids'):
                properties[key]['items'] = {'type':'string'}
            if key == 'approval_id' and tool['name']=='mail.send':
                properties[key]['type'] = ['string','null']
        result.append({'name':tool['name'],'description':tool['description'], 'inputSchema':{'type':'object','properties':properties,'required':tool['required_arguments'],'additionalProperties':False}})
    return result


SCHEMAS = {tool['name']:tool['inputSchema'] for tool in tool_schemas()}
WRITES = frozenset(tool['name'] for tool in CONTRACT['tools'] if tool['category']=='write')


def validate(name, args, operation_id):
    from .dispatcher import ToolError
    schema = SCHEMAS.get(name)
    if schema is None or not isinstance(args,dict) or set(args)-set(schema['properties']) or set(schema['required'])-set(args):
        raise ToolError('INVALID_ARGUMENT')
    if operation_id is not None and (not isinstance(operation_id,str) or not operation_id or len(operation_id)>256):
        raise ToolError('INVALID_ARGUMENT')
    if name in WRITES and not operation_id:
        raise ToolError('INVALID_ARGUMENT')
    types = {'string':str,'integer':int,'array':list,'object':dict,'null':type(None)}
    for key,value in args.items():
        prop = schema['properties'][key]
        kinds = prop['type'] if isinstance(prop['type'],list) else [prop['type']]
        if not any(type(value) is types[kind] for kind in kinds):
            raise ToolError('INVALID_ARGUMENT')
        if prop.get('items') and any(type(item) is not str for item in value):
            raise ToolError('INVALID_ARGUMENT')

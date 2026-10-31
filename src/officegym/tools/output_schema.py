"""Public common ToolResult schema; contains no grader/private fields."""
TOOL_RESULT_SCHEMA = {
    'type':'object', 'required':['ok','data','error','meta'], 'additionalProperties':False,
    'properties':{
        'ok':{'type':'boolean'},
        'data':{'type':['object','null']},
        'error':{'anyOf':[{'type':'null'},{'type':'object','required':['code'],'properties':{'code':{'enum':['INVALID_ARGUMENT','NOT_FOUND','PERMISSION_DENIED','VERSION_CONFLICT','IDEMPOTENCY_CONFLICT','BUDGET_EXHAUSTED']}},'additionalProperties':False}]},
        'meta':{'type':'object','required':['sim_time','observation_id','tool_schema_version'], 'additionalProperties':False,
                'properties':{'sim_time':{'type':'integer'},'observation_id':{'type':'string'},'tool_schema_version':{'const':'tools-v1'}}},
    },
    'allOf':[{'if':{'properties':{'ok':{'const':True}}},'then':{'properties':{'error':{'type':'null'},'data':{'type':'object'}}},'else':{'properties':{'error':{'type':'object'}}}}],
}

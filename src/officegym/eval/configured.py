"""Explicitly budgeted local inference-service evaluation entrypoint."""
import json
from pathlib import Path
from urllib.parse import urlparse
from officegym.harness.loop import Budget
from officegym.harness.policy import ChatPolicy
from officegym.harness.tokenizer import PromptTokenizer
from officegym.harness.context import SimpleContext,FullContext
from officegym.env.world import canonical
from .episode import run_episode


def validate_config(config):
    required = {'endpoint','model','model_version','tokenizer_path','tokenizer_revision','mode','budget','max_context_tokens','max_output_tokens'}
    if required-set(config):
        raise ValueError('missing evaluation configuration: '+','.join(sorted(required-set(config))))
    if any(not isinstance(config[key],str) or not config[key] or config[key].startswith('SET_') for key in ('model','model_version','tokenizer_path','tokenizer_revision')):
        raise ValueError('locked model and tokenizer identities required')
    if config['mode'] not in ('simple','full'):
        raise ValueError('invalid harness mode')
    endpoint = urlparse(config['endpoint'])
    if endpoint.scheme not in ('http','https') or endpoint.hostname not in ('localhost','127.0.0.1','::1') or endpoint.username or endpoint.password:
        raise ValueError('this configuration runner accepts loopback model services only')
    if not {'tool_calls','generated_tokens','wall_seconds','parse_retries'}<=set(config['budget']):
        raise ValueError('all finite budget fields must be explicit')
    budget = Budget(**config['budget'])
    context,output = config['max_context_tokens'],config['max_output_tokens']
    if type(context) is not int or type(output) is not int or not 0<output<context:
        raise ValueError('invalid context/output limits')
    if not Path(config['tokenizer_path']).is_dir():
        raise ValueError('local tokenizer snapshot is missing')
    return budget


def evaluate_instance(instance,out,config_path):
    config = json.loads(Path(config_path).read_text())
    budget = validate_config(config)
    tokenizer = PromptTokenizer.load(config['tokenizer_path'],config['tokenizer_revision'],template_kwargs=config.get('template_kwargs'))
    policy = ChatPolicy(config['endpoint'],config['model'],config['model_version'],max_output_tokens=config['max_output_tokens'],temperature=config.get('temperature',0))
    if config['mode']=='full':
        context = FullContext(tokenizer,config['max_context_tokens'],policy,trigger_tokens=config.get('compression_trigger_tokens',8192),reserved_output_tokens=config['max_output_tokens'],archive_path=Path(out)/'summaries.jsonl')
    else:
        context = SimpleContext(tokenizer,config['max_context_tokens'],config['max_output_tokens'])
    result = run_episode(instance,out,policy,mode=config['mode'],budget=budget,context_manager=context)
    (Path(out)/'tokenizer_manifest.json').write_text(canonical(tokenizer.manifest())+'\n')
    (Path(out)/'resolved_eval_config.json').write_text(canonical(config)+'\n')
    return result

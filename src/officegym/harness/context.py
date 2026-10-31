"""Context selection with tokenizer-provided exact rendered prompt counts."""
import copy
import json
import os
from pathlib import Path
import hashlib
from officegym.env.world import canonical


class ContextOverflow(Exception):
    pass


class SimpleContext:
    def __init__(self,count_tokens,max_tokens,reserved_output_tokens=512):
        if type(max_tokens) is not int or type(reserved_output_tokens) is not int or not 0 <= reserved_output_tokens < max_tokens:
            raise ValueError('invalid context budget')
        self.count_tokens = count_tokens
        self.limit = max_tokens-reserved_output_tokens

    def prepare(self,messages,schemas):
        if len(messages)<2 or messages[0]['role']!='system' or messages[1]['role']!='user':
            raise ValueError('fixed policy and task prefix required')
        prefix = copy.deepcopy(messages[:2])
        groups = []
        index = 2
        while index < len(messages):
            message = messages[index]
            if 'tool_call' in message:
                if index+1 >= len(messages) or messages[index+1].get('role')!='tool' or messages[index+1].get('tool_call_id') != message['tool_call']['id']:
                    raise ValueError('unpaired tool call')
                groups.append(copy.deepcopy(messages[index:index+2]))
                index += 2
            else:
                if message.get('role')=='tool':
                    raise ValueError('orphan tool result')
                groups.append([copy.deepcopy(message)])
                index += 1
        removed = []
        original_tokens = None
        while True:
            selected = prefix + [message for group in groups for message in group]
            tokens = self.count_tokens(selected,schemas)
            if type(tokens) is not int or tokens<0:
                raise ValueError('token counter must return nonnegative actual prompt count')
            if original_tokens is None:
                original_tokens = tokens
            if tokens <= self.limit:
                return selected, {'context_metrics_version':'context-metrics-v1','source_input_tokens':original_tokens,'input_tokens':tokens,'removed_groups':len(removed),
                                  'removed_sha256':hashlib.sha256(canonical(removed).encode()).hexdigest() if removed else None}
            if not groups:
                raise ContextOverflow('policy, task and tool schemas exceed context budget')
            removed.append(groups.pop(0))


def observed_refs(messages):
    refs = set()
    def collect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if (key=='id' or key.endswith('_id')) and isinstance(item,str):
                    refs.add(item)
                    if type(value.get('version')) is int:
                        refs.add(item+'/'+str(value['version']))
                if key=='seq' and type(item) is int:
                    refs.add('event_seq:'+str(item))
                collect(item)
        elif isinstance(value,list):
            for item in value:
                collect(item)
    for message in messages:
        if message.get('role')!='tool':
            continue
        try:
            collect(json.loads(message.get('content','null')))
        except (ValueError,TypeError):
            continue
    return refs


def grounded_summary(summary,required,source):
    if not isinstance(summary,dict) or not required <= set(summary) or not all(isinstance(summary[key],list) for key in required):
        return False
    allowed = observed_refs(source)
    for item in summary['evidence']+summary['confirmed']:
        if not isinstance(item,dict) or not isinstance(item.get('source_refs'),list) or not item['source_refs']:
            return False
        if any(not isinstance(ref,str) or ref not in allowed for ref in item['source_refs']):
            return False
    return True


class FullContext(SimpleContext):
    """Summarizer must be the executing policy, with measured usage and provenance."""
    REQUIRED = {'goals','constraints','evidence','completed','blockers','waiting','unknown_operations','confirmed','hypotheses'}

    def __init__(self,count_tokens,max_tokens,policy,trigger_tokens=8192,reserved_output_tokens=512,archive_path=None):
        super().__init__(count_tokens,max_tokens,reserved_output_tokens)
        self.policy,self.trigger = policy,trigger_tokens
        self.cache = {}
        self.remaining = None
        self.archive_path = Path(archive_path) if archive_path else None
        self.archive_errors = 0
        if self.archive_path and self.archive_path.exists():
            for line in self.archive_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                    if hashlib.sha256(canonical(item['record']).encode()).hexdigest()!=item['record_sha256']:
                        raise ValueError('summary archive checksum mismatch')
                    self.cache[item['source_sha256']] = item['record']
                except (ValueError,KeyError,TypeError):
                    self.archive_errors += 1


    def set_budget(self,remaining):
        self.remaining = dict(remaining)

    def prepare(self,messages,schemas):
        # Validate pairs before calling the model; truncation is the failure fallback.
        fallback,fallback_info = super().prepare(messages,schemas)
        count = self.count_tokens(messages,schemas)
        if count <= self.trigger or len(messages)<=4:
            return fallback,fallback_info
        fallback_info = dict(fallback_info,compression_triggered=True)
        source = messages[2:-2]
        if source and source[-1].get('tool_call'):
            return fallback,dict(fallback_info,compression_failed='unsafe_boundary')
        fingerprint = hashlib.sha256(canonical({'source':source,'prefix':messages[:2],'schemas':schemas,'model_version':getattr(self.policy,'model_version',None)}).encode()).hexdigest()
        usage = {'summary_input_tokens':0,'summary_output_tokens':0,'summary_model_call':False,'summary_cache_hit':False}
        if fingerprint in self.cache:
            record = self.cache[fingerprint]
            usage['summary_cache_hit'] = True
        else:
            if self.remaining is None:
                raise ValueError('compression requires remaining global budget')
            record = self.policy.summarize(copy.deepcopy(source),dict(self.remaining))
            for field in ('input_tokens','output_tokens'):
                if type(record.get(field)) is not int or record[field]<0:
                    raise ValueError('summary must report measured token usage')
            usage = dict(summary_input_tokens=record['input_tokens'],summary_output_tokens=record['output_tokens'],summary_raw=record.get('raw'),summary_model_call=True,summary_cache_hit=False)
            summary = record.get('summary',{})
            valid = grounded_summary(summary,self.REQUIRED,source)
            if not valid:
                return fallback,dict(fallback_info,**usage,compression_failed='invalid_summary',source_sha256=fingerprint)
            if not record.get('model_version') or not isinstance(record.get('sampling_parameters'),dict):
                raise ValueError('summary model provenance required')
            self.cache[fingerprint] = copy.deepcopy(record)
            if self.archive_path:
                self.archive_path.parent.mkdir(parents=True,exist_ok=True)
                item = {'source_sha256':fingerprint,'record':record,'record_sha256':hashlib.sha256(canonical(record).encode()).hexdigest()}
                with self.archive_path.open('a') as stream:
                    stream.write(canonical(item)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
        if not grounded_summary(record.get('summary'),self.REQUIRED,source):
            return fallback,dict(fallback_info,**usage,compression_failed='invalid_archived_summary',source_sha256=fingerprint)
        selected = copy.deepcopy(messages[:2])+[{'role':'user','content':canonical({'kind':'model_summary_not_ground_truth','source_sha256':fingerprint,'summary':record['summary']})}]+copy.deepcopy(messages[-2:])
        tokens = self.count_tokens(selected,schemas)
        if tokens>self.limit:
            return fallback,dict(fallback_info,**usage,compression_failed='summary_too_large',source_sha256=fingerprint)
        return selected,dict(context_metrics_version='context-metrics-v1',source_input_tokens=count,compression_triggered=True,input_tokens=tokens,compressed=True,source_sha256=fingerprint,summary=record['summary'],summary_model_version=record['model_version'],sampling_parameters=record['sampling_parameters'],**usage)

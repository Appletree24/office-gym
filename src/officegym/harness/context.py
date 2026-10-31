"""Context selection with tokenizer-provided exact rendered prompt counts."""
import copy
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
        while True:
            selected = prefix + [message for group in groups for message in group]
            tokens = self.count_tokens(selected,schemas)
            if type(tokens) is not int or tokens<0:
                raise ValueError('token counter must return nonnegative actual prompt count')
            if tokens <= self.limit:
                return selected, {'input_tokens':tokens,'removed_groups':len(removed),
                                  'removed_sha256':hashlib.sha256(canonical(removed).encode()).hexdigest() if removed else None}
            if not groups:
                raise ContextOverflow('policy, task and tool schemas exceed context budget')
            removed.append(groups.pop(0))

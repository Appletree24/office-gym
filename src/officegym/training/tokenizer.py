"""Strict SFT alignment using the locked tokenizer's native assistant mask."""
from officegym.harness.policy import render_messages
from .data import RejectedTrajectory,verify_mask


class SFTTokenizer:
    def __init__(self,tokenizer,template_kwargs=None):
        self.tokenizer = tokenizer
        self.kwargs = dict(template_kwargs or {})
        if {'tokenize','tools','return_dict','return_assistant_tokens_mask','add_generation_prompt','truncation'} & set(self.kwargs):
            raise ValueError('cannot override SFT tokenization invariants')

    def __call__(self,messages,tools):
        rendered = render_messages(messages)
        schemas = [{'type':'function','function':{'name':tool['name'],'description':tool['description'],'parameters':tool['inputSchema']}} for tool in tools]
        encoded = self.tokenizer.apply_chat_template(rendered,tools=schemas,tokenize=True,add_generation_prompt=False,return_dict=True,return_assistant_tokens_mask=True,**self.kwargs)
        ids,mask = encoded['input_ids'],encoded.get('assistant_masks')
        if not isinstance(ids,list) or not isinstance(mask,list) or len(ids)!=len(mask):
            raise RejectedTrajectory('template_missing_assistant_mask')
        roles = ['template']*len(ids)
        previous = 0
        for index,message in enumerate(rendered):
            prefix = self.tokenizer.apply_chat_template(rendered[:index+1],tools=schemas,tokenize=True,add_generation_prompt=False,**self.kwargs)
            if not isinstance(prefix,list) or len(prefix)<previous or prefix!=ids[:len(prefix)]:
                raise RejectedTrajectory('template_prefix_not_token_stable')
            for position in range(previous,len(prefix)):
                roles[position] = message['role']
            previous = len(prefix)
        if previous!=len(ids):
            raise RejectedTrajectory('template_token_boundary_mismatch')
        verify_mask(ids,mask,roles)
        # Every assistant message must actually supply a supervised span.
        previous = 0
        for index,message in enumerate(rendered):
            prefix = self.tokenizer.apply_chat_template(rendered[:index+1],tools=schemas,tokenize=True,add_generation_prompt=False,**self.kwargs)
            if message['role']=='assistant' and not any(mask[previous:len(prefix)]):
                raise RejectedTrajectory('assistant_message_has_no_loss')
            previous = len(prefix)
        return {'input_ids':ids,'assistant_mask':mask,'token_roles':roles}

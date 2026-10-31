"""Strict SFT alignment using native masks and source-locked annotations."""
from officegym.harness.policy import render_template_messages
from .data import RejectedTrajectory,verify_mask
from .mask_templates import annotated_template,ADAPTER_VERSION
import hashlib


class SFTTokenizer:
    def __init__(self,tokenizer,template_kwargs=None):
        self.tokenizer = tokenizer
        self.kwargs = dict(template_kwargs or {})
        if {'tokenize','tools','return_dict','return_assistant_tokens_mask','add_generation_prompt','truncation','chat_template'} & set(self.kwargs):
            raise ValueError('cannot override SFT tokenization invariants')
        self.source_template=getattr(tokenizer,'chat_template',None)
        self.mask_template=annotated_template(self.source_template)

    def mask_manifest(self):
        source=self.source_template
        adapted=self.mask_template or source
        return dict(message_normalization='decoded-canonical-tool-arguments-v1',strategy=ADAPTER_VERSION if self.mask_template else 'native_generation_mask',source_template_sha256=hashlib.sha256(source.encode()).hexdigest() if isinstance(source,str) else None,mask_template_sha256=hashlib.sha256(adapted.encode()).hexdigest() if isinstance(adapted,str) else None)

    def __call__(self,messages,tools):
        if self.mask_template and getattr(self.tokenizer,'chat_template',None)!=self.source_template:
            raise RejectedTrajectory('mask_annotation_source_changed')
        rendered = render_template_messages(messages)
        schemas = [{'type':'function','function':{'name':tool['name'],'description':tool['description'],'parameters':tool['inputSchema']}} for tool in tools]
        mask_kwargs=dict(self.kwargs)
        if self.mask_template:
            mask_kwargs['chat_template']=self.mask_template
            original_text=self.tokenizer.apply_chat_template(rendered,tools=schemas,tokenize=False,add_generation_prompt=False,**self.kwargs)
            adapted_text=self.tokenizer.apply_chat_template(rendered,tools=schemas,tokenize=False,add_generation_prompt=False,**mask_kwargs)
            if original_text!=adapted_text:
                raise RejectedTrajectory('mask_annotation_changed_rendered_text')
        encoded = self.tokenizer.apply_chat_template(rendered,tools=schemas,tokenize=True,add_generation_prompt=False,return_dict=True,return_assistant_tokens_mask=True,**mask_kwargs)
        ids,mask = encoded['input_ids'],encoded.get('assistant_masks')
        if not isinstance(ids,list) or not isinstance(mask,list) or len(ids)!=len(mask):
            raise RejectedTrajectory('template_missing_assistant_mask')
        if self.mask_template and ids!=self.tokenizer.apply_chat_template(rendered,tools=schemas,tokenize=True,add_generation_prompt=False,**self.kwargs):
            raise RejectedTrajectory('mask_annotation_changed_token_ids')
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

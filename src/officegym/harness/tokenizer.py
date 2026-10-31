"""Exact prompt sizing with the same locked tokenizer/template as inference."""
import hashlib
import json
from .policy import render_template_messages


class PromptTokenizer:
    def __init__(self,tokenizer,*,model_revision,template_kwargs=None):
        if not model_revision or not getattr(tokenizer,'chat_template',None):
            raise ValueError('locked revision and explicit chat template required')
        self.tokenizer,self.model_revision = tokenizer,model_revision
        self.template_kwargs = dict(template_kwargs or {})
        forbidden = {'tokenize','add_generation_prompt','tools','return_tensors','return_dict','chat_template'}
        if forbidden & set(self.template_kwargs):
            raise ValueError('template kwargs cannot override sizing contract')
        template = tokenizer.chat_template
        encoded = template if isinstance(template,str) else json.dumps(template,sort_keys=True,ensure_ascii=False)
        self.template_sha256 = hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def load(cls,path,revision,**kwargs):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path,revision=revision,trust_remote_code=False,local_files_only=True)
        return cls(tokenizer,model_revision=revision,**kwargs)

    def token_ids(self,messages,schemas):
        tools = [{'type':'function','function':{'name':tool['name'],'description':tool['description'],'parameters':tool['inputSchema']}} for tool in schemas]
        ids = self.tokenizer.apply_chat_template(render_template_messages(messages),tools=tools,tokenize=True,add_generation_prompt=True,return_dict=False,**self.template_kwargs)
        if not isinstance(ids,list) or any(type(token) is not int for token in ids):
            raise ValueError('tokenizer must return one original flat ID sequence')
        return ids

    def __call__(self,messages,schemas):
        return len(self.token_ids(messages,schemas))

    def manifest(self):
        vocabulary = self.tokenizer.get_vocab()
        vocab_hash = hashlib.sha256(json.dumps(vocabulary,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        return {'model_revision':self.model_revision,'chat_template_sha256':self.template_sha256,'vocabulary_sha256':vocab_hash,'template_kwargs':self.template_kwargs,'message_normalization':'decoded-canonical-tool-arguments-v1'}

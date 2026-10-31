import unittest
from officegym.training.tokenizer import SFTTokenizer
from officegym.training.data import RejectedTrajectory


class SFTTokenizerTests(unittest.TestCase):
    def test_native_mask_independently_checked_against_roles(self):
        class Tokenizer:
            corrupt = False
            def apply_chat_template(self,messages,**kwargs):
                ids = list(range(len(messages)*2))
                if kwargs.get('return_dict',True):
                    mask = [int(message['role']=='assistant' or self.corrupt) for message in messages for _ in range(2)]
                    return {'input_ids':ids,'assistant_masks':mask}
                return ids
        tokenizer = Tokenizer()
        messages = [{'role':'user','content':'task'},{'role':'assistant','tool_call':{'name':'board.read','arguments':{},'id':'one'}},{'role':'tool','tool_call_id':'one','content':'result'}]
        encoded = SFTTokenizer(tokenizer)(messages,[])
        self.assertEqual(encoded['assistant_mask'],[0,0,1,1,0,0])
        self.assertEqual(encoded['token_roles'],['user','user','assistant','assistant','tool','tool'])
        tokenizer.corrupt = True
        with self.assertRaisesRegex(RejectedTrajectory,'non_actor_token_has_loss'):
            SFTTokenizer(tokenizer)(messages,[])

    def test_annotation_must_preserve_text_and_original_ids(self):
        from unittest.mock import patch
        class Tokenizer:
            chat_template='original fixture template'
            corrupt_text=False
            corrupt_ids=False
            def apply_chat_template(self,messages,**kwargs):
                if not kwargs['tokenize']:
                    return 'changed' if kwargs.get('chat_template') and self.corrupt_text else 'same authored rendering'
                ids=list(range(len(messages)*2))
                if kwargs.get('return_dict',True):
                    return dict(input_ids=[99]+ids[1:] if self.corrupt_ids else ids,assistant_masks=[int(m['role']=='assistant') for m in messages for _ in range(2)])
                return ids
        messages=[dict(role='user',content='task'),dict(role='assistant',content='answer')]
        tokenizer=Tokenizer()
        with patch('officegym.training.tokenizer.annotated_template',return_value='annotated fixture'):
            adapter=SFTTokenizer(tokenizer)
        self.assertEqual(adapter(messages,[])['input_ids'],[0,1,2,3])
        tokenizer.corrupt_text=True
        with self.assertRaisesRegex(RejectedTrajectory,'changed_rendered_text'):
            adapter(messages,[])
        tokenizer.corrupt_text=False
        tokenizer.corrupt_ids=True
        with self.assertRaisesRegex(RejectedTrajectory,'changed_token_ids'):
            adapter(messages,[])
        tokenizer.chat_template='changed source fixture'
        with self.assertRaisesRegex(RejectedTrajectory,'source_changed'):
            adapter(messages,[])

    def test_only_exact_source_template_can_be_annotated(self):
        import hashlib
        from unittest.mock import patch
        from officegym.training.mask_templates import annotated_template
        self.assertIsNone(annotated_template('unknown template'))
        with patch('officegym.training.mask_templates.QWEN3_2507_SOURCE_SHA256',hashlib.sha256(b'missing boundaries').hexdigest()):
            with self.assertRaisesRegex(ValueError,'boundary mismatch'):
                annotated_template('missing boundaries')
        with self.assertRaisesRegex(ValueError,'invariants'):
            SFTTokenizer(object(),template_kwargs={'chat_template':'changed'})

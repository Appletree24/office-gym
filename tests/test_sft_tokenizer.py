import unittest
from officegym.training.tokenizer import SFTTokenizer
from officegym.training.data import RejectedTrajectory


class SFTTokenizerTests(unittest.TestCase):
    def test_native_mask_independently_checked_against_roles(self):
        class Tokenizer:
            corrupt = False
            def apply_chat_template(self,messages,**kwargs):
                ids = list(range(len(messages)*2))
                if kwargs.get('return_dict'):
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

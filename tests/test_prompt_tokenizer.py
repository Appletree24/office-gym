import unittest
from officegym.harness.tokenizer import PromptTokenizer


class PromptTokenizerTests(unittest.TestCase):
    def test_renders_tools_and_preserves_actual_ids(self):
        class Tokenizer:
            chat_template = 'locked template'
            captured = None
            def apply_chat_template(self,messages,**kwargs):
                self.captured = (messages,kwargs)
                return [10,20,30,40,50]
            def get_vocab(self):
                return {'tool':10,'result':20}
        tokenizer = Tokenizer()
        counter = PromptTokenizer(tokenizer,model_revision='commit123')
        messages = [{'role':'assistant','tool_call':{'id':'call1','name':'board.read','arguments':{}}},{'role':'tool','tool_call_id':'call1','content':'{}'}]
        schemas = [{'name':'board.read','description':'Read','inputSchema':{'type':'object'}}]
        self.assertEqual(counter(messages,schemas),5)
        self.assertEqual(counter.token_ids(messages,schemas),[10,20,30,40,50])
        rendered,kwargs = tokenizer.captured
        self.assertEqual(rendered[0]['tool_calls'][0]['function']['arguments'],{})
        self.assertTrue(kwargs['tokenize'])
        self.assertTrue(kwargs['add_generation_prompt'])
        self.assertEqual(kwargs['tools'][0]['function']['name'],'board.read')
        self.assertEqual(counter.manifest()['model_revision'],'commit123')

    def test_template_override_rejected(self):
        class Tokenizer:
            chat_template = 'template'
        for key in ('tokenize','chat_template'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                PromptTokenizer(Tokenizer(),model_revision='revision',template_kwargs={key:False})

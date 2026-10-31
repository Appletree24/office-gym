import unittest
from officegym.harness.context import SimpleContext,ContextOverflow


class ContextTests(unittest.TestCase):
    def test_whole_pairs_and_original_preserved(self):
        messages = [{'role':'system','content':'policy'},{'role':'user','content':'task'}]
        for i in range(3):
            messages.extend([{'role':'assistant','tool_call':{'id':str(i),'name':'board.read','arguments':{}}}, {'role':'tool','tool_call_id':str(i),'content':'result'}])
        # Synthetic counter tests selection only, not real model context sizes.
        context = SimpleContext(lambda messages,schemas:len(messages)*10,65,5)
        selected,info = context.prepare(messages,[])
        self.assertEqual(len(selected),6)
        self.assertEqual(info['removed_groups'],1)
        self.assertEqual(selected[2]['tool_call']['id'],'1')
        self.assertEqual(len(messages),8)
        self.assertEqual(selected[:2],messages[:2])

    def test_overflow_and_invalid_pair(self):
        context = SimpleContext(lambda messages,schemas:100,60,5)
        with self.assertRaises(ContextOverflow):
            context.prepare([{'role':'system'},{'role':'user'}],[])
        with self.assertRaises(ValueError):
            context.prepare([{'role':'system'},{'role':'user'},{'role':'assistant','tool_call':{'id':'1'}}],[])

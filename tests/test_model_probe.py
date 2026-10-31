import unittest
from officegym.eval.model_probe import parse_tool
from officegym.tools.dispatcher import ToolError


class ModelProbeParserTests(unittest.TestCase):
    def test_native_call_with_rationale_and_stop_token(self):
        value=parse_tool('I will read it.\n<tool_call>\n{"name":"board.read","arguments":{}}\n</tool_call><|im_end|>')
        self.assertEqual(value,dict(name='board.read',arguments={}))

    def test_multiple_missing_and_non_object_calls_are_rejected(self):
        cases=['<tool_call>{"name":"board.read","arguments":{}}</tool_call><tool_call>', 'plain text','<tool_call>{"name":"board.read","arguments":{}}</tool_call>'*2,'<tool_call>{"name":"board.read","arguments":[]}</tool_call>','<tool_call>{"name":"board.read","arguments":{},"extra":true}</tool_call>']
        for text in cases:
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_tool(text)

    def test_business_schema_is_still_enforced(self):
        for text in ('<tool_call>{"name":"unknown.tool","arguments":{}}</tool_call>','<tool_call>{"name":"board.read","arguments":{"hidden":true}}</tool_call>'):
            with self.subTest(text=text),self.assertRaises(ToolError):
                parse_tool(text)

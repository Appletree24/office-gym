import unittest
from officegym.harness.policy import ChatPolicy,render_messages


class PolicyTests(unittest.TestCase):
    def test_request_cap_and_parse_preserves_usage(self):
        captured = []
        def transport(payload,timeout):
            captured.append((payload,timeout))
            return {'choices':[{'message':{'tool_calls':[{'function':{'name':'board.read','arguments':'{}'}}]}}],'usage':{'prompt_tokens':20,'completion_tokens':4}}
        policy = ChatPolicy('http://localhost:8000/v1/chat/completions','student','revision',transport=transport)
        turn = policy.generate([{'role':'user','content':'task'}],[],{'generated_tokens':8,'wall_seconds':12})
        self.assertEqual(turn.tool,'board.read')
        self.assertEqual(turn.arguments,{})
        self.assertEqual(turn.output_tokens,4)
        self.assertEqual(captured[0][0]['max_tokens'],8)
        self.assertFalse(captured[0][0]['parallel_tool_calls'])

    def test_missing_usage_rejected(self):
        policy = ChatPolicy('local','student','revision',transport=lambda *args:{'choices':[]})
        with self.assertRaises(ValueError):
            policy.generate([],[],{'generated_tokens':8,'wall_seconds':12})

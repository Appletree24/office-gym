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

    def test_malformed_envelopes_preserve_measured_usage_and_raw_response(self):
        malformed=[{}, {'choices':[]}, {'choices':[None]}, {'choices':[{'message':None}]},
                   {'choices':[{'message':{'tool_calls':[None]}}]},
                   {'choices':[{'message':{'tool_calls':[{'function':None}]}}]}]
        for response in malformed:
            response['usage']={'prompt_tokens':9,'completion_tokens':7}
            policy=ChatPolicy('local','student','revision',transport=lambda *args:response)
            turn=policy.generate([],[],{'generated_tokens':20,'wall_seconds':10})
            self.assertIsNone(turn.tool)
            self.assertEqual((turn.input_tokens,turn.output_tokens),(9,7))
            self.assertEqual(turn.raw,response)
            summary=policy.summarize([],{'generated_tokens':20,'wall_seconds':10})
            self.assertEqual(summary['output_tokens'],7)
            self.assertEqual(summary['raw'],response)

    def test_malformed_response_is_bounded_parse_failure_with_full_usage(self):
        import json
        import tempfile
        from pathlib import Path
        from officegym.harness.loop import Harness,Budget
        response={'choices':[],'usage':{'prompt_tokens':9,'completion_tokens':7}}
        policy=ChatPolicy('fixture','student','fixture-revision',transport=lambda *args:response)
        with tempfile.TemporaryDirectory() as directory:
            trace=Path(directory)/'trace.jsonl'
            result=Harness(policy,lambda *args:self.fail('malformed response executed tool'),{},trace,Budget(parse_retries=2)).run()
            self.assertEqual(result['state'],'PARSE_ERROR')
            self.assertEqual(result['generated_tokens'],21)
            self.assertEqual(result['input_tokens'],27)
            rows=[json.loads(line) for line in trace.read_text().splitlines()]
            model_rows=[row for row in rows if row['kind']=='model_response']
            self.assertEqual(len(model_rows),3)
            self.assertTrue(all(row['raw']==response for row in model_rows))

    def test_assistant_text_survives_tool_message_rendering(self):
        response={'choices':[{'message':{'content':'I will finish now.','tool_calls':[{'function':{'name':'task.finish','arguments':'{"summary":"done"}'}}]}}],'usage':{'prompt_tokens':9,'completion_tokens':7}}
        policy=ChatPolicy('fixture','student','fixture',transport=lambda *args:response)
        turn=policy.generate([],[],{'generated_tokens':20,'wall_seconds':10})
        self.assertEqual(turn.content,'I will finish now.')
        message={'role':'assistant','content':turn.content,'tool_call':{'id':'1','name':turn.tool,'arguments':turn.arguments}}
        self.assertEqual(render_messages([message])[0]['content'],'I will finish now.')

"""Protocol translation preserves the model-selected action and observed values."""
import copy,json,unittest
from officegym.training.provider_protocols import encode_single_action,decode_single_action


class ProviderProtocolTests(unittest.TestCase):
    def test_one_json_action_roundtrip_and_original_response_preservation(self):
        payload=dict(model='fixture',tools=[dict(type='function',function=dict(name='board.read',parameters={'type':'object'}))],messages=[dict(role='system',content='original policy'),dict(role='assistant',content=None,tool_calls=[dict(id='1',function=dict(name='board.read',arguments='{}'))]),dict(role='tool',tool_call_id='1',content='{"actual":"observation"}')],tool_choice='required',parallel_tool_calls=False)
        before=copy.deepcopy(payload);encoded,names=encode_single_action(payload)
        self.assertEqual(payload,before);self.assertNotIn('tools',encoded);self.assertEqual(encoded['response_format'],{'type':'json_object'});self.assertIn('"actual":"observation"',encoded['messages'][-1]['content'])
        raw=dict(choices=[dict(message=dict(content='{"name":"board.read","arguments":{}}'))]);result=decode_single_action(raw,names)
        self.assertEqual(raw['choices'][0]['message']['content'],'{"name":"board.read","arguments":{}}');self.assertIsNone(result['choices'][0]['message']['content']);self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'],'board.read')

    def test_multiple_or_unknown_actions_are_not_silently_selected(self):
        for action in ([dict(name='board.read',arguments={})],dict(name='unknown',arguments={}),dict(name='board.read',arguments={},more_actions=[])):
            raw=dict(choices=[dict(message=dict(content=json.dumps(action)))]);self.assertEqual(decode_single_action(raw,{'board.read'}),raw)

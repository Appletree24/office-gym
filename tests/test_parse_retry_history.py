from pathlib import Path
import tempfile
import unittest
from officegym.harness.loop import Harness,Budget,PARSE_FEEDBACK
from officegym.harness.policy import ChatPolicy


class ParseRetryHistoryTests(unittest.TestCase):
    def test_invalid_reply_is_retained_as_inert_history_and_never_executes_partial_batch(self):
        requests=[];calls=[]
        messages=[{'content':'Not a tool'}, {'tool_calls':[{'function':{'name':'board.patch','arguments':'{}'}},{'function':{'name':'board.read','arguments':'{}'}}]}, {'tool_calls':[{'function':{'name':'task.finish','arguments':'{"summary":"done"}'}}]}]
        def transport(payload,timeout):
            requests.append(payload)
            return {'choices':[{'message':messages[len(requests)-1]}],'usage':{'prompt_tokens':9,'completion_tokens':7}}
        policy=ChatPolicy('fixture','fixture','fixture-revision',transport=transport)
        def call(name,args,key):
            calls.append(name);return {'ok':True,'data':{},'error':None,'meta':{}}
        with tempfile.TemporaryDirectory() as directory:
            result=Harness(policy,call,{},Path(directory)/'trace.jsonl',Budget(parse_retries=2)).run()
        self.assertEqual(result['state'],'DECLARED_DONE');self.assertEqual(result['generated_tokens'],21);self.assertEqual(calls,['task.finish'])
        second=requests[1]['messages'];self.assertEqual(second[-2],{'role':'assistant','content':'Not a tool'});self.assertEqual(second[-1]['content'],PARSE_FEEDBACK)
        failed=requests[2]['messages'][-2];self.assertEqual(failed['role'],'assistant');self.assertIn('board.patch',failed['content']);self.assertNotIn('tool_calls',failed)

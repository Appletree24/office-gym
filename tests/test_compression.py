import unittest
from officegym.harness.context import FullContext


class CompressionTests(unittest.TestCase):
    def test_structured_summary_cost_and_cache(self):
        class Policy:
            def summarize(self,messages,budget):
                return dict(summary={key:[] for key in FullContext.REQUIRED},input_tokens=30,output_tokens=12,model_version='test-policy',sampling_parameters={'temperature':0})
        messages = [{'role':'system'},{'role':'user'}]
        for i in range(4):
            messages += [{'role':'assistant','tool_call':{'id':str(i)}},{'role':'tool','tool_call_id':str(i)}]
        context = FullContext(lambda messages,schemas:len(messages)*10,200,Policy(),trigger_tokens=50,reserved_output_tokens=10)
        context.set_budget({'generated_tokens':100})
        selected,info = context.prepare(messages,[])
        self.assertEqual(len(selected),5)
        self.assertTrue(info['compressed'])
        self.assertEqual(info['summary_output_tokens'],12)
        cached=context.prepare(messages,[])[1]
        self.assertEqual(cached['summary_output_tokens'],0)
        self.assertTrue(info['compression_triggered'])
        self.assertTrue(info['summary_model_call'])
        self.assertFalse(info['summary_cache_hit'])
        self.assertTrue(cached['summary_cache_hit'])
        self.assertFalse(cached['summary_model_call'])

    def test_archive_reused_after_reconstruction(self):
        import tempfile
        from pathlib import Path
        class Policy:
            model_version = 'same-model'
            calls = 0
            def summarize(self,messages,budget):
                self.calls += 1
                return dict(summary={key:[] for key in FullContext.REQUIRED},input_tokens=30,output_tokens=12,model_version=self.model_version,sampling_parameters={})
        policy = Policy()
        messages = [{'role':'system'},{'role':'user'}]
        for i in range(3):
            messages += [{'role':'assistant','tool_call':{'id':str(i)}},{'role':'tool','tool_call_id':str(i)}]
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'summaries.jsonl'
            first = FullContext(lambda messages,schemas:len(messages)*10,200,policy,50,10,archive)
            first.set_budget({'generated_tokens':100})
            original = first.prepare(messages,[])[0]
            second = FullContext(lambda messages,schemas:len(messages)*10,200,policy,50,10,archive)
            second.set_budget({'generated_tokens':100})
            restored,info = second.prepare(messages,[])
            self.assertEqual(restored,original)
            self.assertEqual(info['summary_output_tokens'],0)
            self.assertEqual(policy.calls,1)

    def test_invented_evidence_rejected_with_cost_retained(self):
        import json
        class Policy:
            def summarize(self,messages,budget):
                summary = {key:[] for key in FullContext.REQUIRED}
                summary['confirmed'] = [{'fact':'approved','source_refs':['not_observed']}]
                return dict(summary=summary,input_tokens=20,output_tokens=10,model_version='fixture',sampling_parameters={})
        messages = [{'role':'system'},{'role':'user'}]
        for i in range(3):
            messages += [{'role':'assistant','tool_call':{'id':str(i)}},{'role':'tool','tool_call_id':str(i),'content':json.dumps({'meta':{'observation_id':'obs_'+str(i)}})}]
        context = FullContext(lambda messages,schemas:len(messages)*10,200,Policy(),50,10)
        context.set_budget({'generated_tokens':100})
        selected,info = context.prepare(messages,[])
        self.assertTrue(info['compression_triggered'])
        self.assertTrue(info['summary_model_call'])
        self.assertEqual(info['compression_failed'],'invalid_summary')
        self.assertEqual(info['summary_output_tokens'],10)
        self.assertEqual(selected,messages)

    def test_oversized_summary_request_is_not_sent(self):
        class Policy:
            def summarize(self,*args):
                raise AssertionError('oversized summary request sent')
        def counter(messages,schemas):
            return 1000 if messages[0].get('content','').startswith('Summarize prior') else len(messages)*10
        messages=[{'role':'system'},{'role':'user'}]
        for index in range(3):
            messages += [{'role':'assistant','tool_call':{'id':str(index)}},{'role':'tool','tool_call_id':str(index)}]
        context=FullContext(counter,200,Policy(),50,10)
        context.set_budget({'generated_tokens':100})
        selected,info=context.prepare(messages,[])
        self.assertEqual(info['compression_failed'],'summary_context_overflow')
        self.assertFalse(info['summary_model_call'])
        self.assertEqual(selected,messages)

    def test_incremental_summary_requires_exact_prior_source_prefix(self):
        import copy
        class Policy:
            calls=[]
            def summarize(self,messages,budget):
                self.calls.append(copy.deepcopy(messages))
                return dict(summary={key:[] for key in FullContext.REQUIRED},input_tokens=30,output_tokens=12,model_version='fixture',sampling_parameters={})
        messages=[{'role':'system'},{'role':'user'}]
        for index in range(3):
            messages += [{'role':'assistant','tool_call':{'id':str(index)}},{'role':'tool','tool_call_id':str(index),'content':'{}'}]
        policy=Policy();context=FullContext(lambda messages,schemas:len(messages)*10,200,policy,50,10)
        context.set_budget({'generated_tokens':100})
        context.prepare(messages,[])
        extended=messages+[{'role':'assistant','tool_call':{'id':'3'}},{'role':'tool','tool_call_id':'3','content':'{}'}]
        context.prepare(extended,[])
        self.assertEqual(len(policy.calls[0]),4)
        self.assertEqual(len(policy.calls[1]),3)
        self.assertIn('previous_model_summary_not_ground_truth',policy.calls[1][0]['content'])
        changed=copy.deepcopy(extended)
        changed[3]['content']='{"changed":true}'
        context.prepare(changed,[])
        self.assertEqual(len(policy.calls[2]),6)
        self.assertIn('tool_call',policy.calls[2][0])

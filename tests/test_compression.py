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
        self.assertEqual(context.prepare(messages,[])[1]['summary_output_tokens'],0)

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

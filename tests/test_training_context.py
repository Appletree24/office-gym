"""Authored counters exercise strict history boundaries, not model lengths."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.harness.context import TrainingContext,ContextOverflow
from officegym.harness.loop import Harness,AssistantTurn


class TrainingContextTests(unittest.TestCase):
    def test_reserve_output_and_keep_entire_history(self):
        messages=[dict(role='system',content='s'),dict(role='user',content='u'),dict(role='assistant',content='a')]
        context=TrainingContext(lambda m,s:90,100,10,count_history_tokens=lambda m,s:100)
        selected,info=context.prepare(messages,[])
        self.assertEqual(selected,messages);self.assertIsNot(selected,messages)
        self.assertEqual(info['removed_groups'],0);self.assertFalse(info['compressed'])
        self.assertEqual(context.validate_history(messages,[])['full_history_tokens'],100)
        context.count_tokens=lambda m,s:91
        with self.assertRaises(ContextOverflow):context.prepare(messages,[])
        self.assertEqual(len(messages),3)
        with self.assertRaises(ValueError):TrainingContext(lambda m,s:0,8193,10,count_history_tokens=lambda m,s:0)

    def test_final_tool_observation_overflow_prevents_declared_done(self):
        class Policy:
            calls=0
            def generate(self,*args):
                self.calls+=1;return AssistantTurn('task.finish',{},50,4)
        for history,expected in ((100,'DECLARED_DONE'),(101,'CONTEXT_OVERFLOW')):
            with self.subTest(history=history),tempfile.TemporaryDirectory() as directory:
                context=TrainingContext(lambda m,s:50,100,10,count_history_tokens=lambda m,s:history)
                policy=Policy();trace=Path(directory)/'trace'
                harness=Harness(policy,lambda *args:dict(ok=True,data={'large':'observation'},error=None),{},trace,context_manager=context)
                result=harness.run();self.assertEqual(result['state'],expected);self.assertEqual(result['tool_calls'],1)
                rows=[json.loads(line) for line in trace.read_text().splitlines()]
                history_row=next(row for row in rows if row['kind'].startswith('context_history_'))
                self.assertEqual(history_row['full_history_tokens'],history)
                self.assertEqual(policy.calls,1);self.assertEqual(len(harness.messages),4)

    def test_overflow_before_generation_has_no_model_or_tool_calls(self):
        class Policy:
            def generate(self,*args):raise AssertionError('must not generate')
        with tempfile.TemporaryDirectory() as directory:
            context=TrainingContext(lambda m,s:91,100,10,count_history_tokens=lambda m,s:0)
            harness=Harness(Policy(),lambda *args:self.fail('must not execute'),{},Path(directory)/'trace',context_manager=context)
            result=harness.run();self.assertEqual(result['state'],'CONTEXT_OVERFLOW');self.assertEqual(result['generated_tokens'],0);self.assertEqual(result['tool_calls'],0)

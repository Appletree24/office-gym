"""Authored alignment viewer fixtures; no human or teacher proof."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import canonical
from officegym.training.mask_review import prepare,digest


class FixtureTokenizer:
    tokenizer=None
    def __init__(self):self.tokenizer=self
    def mask_manifest(self):return dict(strategy='software-fixture')
    def __call__(self,messages,tools):return dict(input_ids=[1,2,3],assistant_mask=[0,1,0],token_roles=['user','assistant','tool'])
    def convert_ids_to_tokens(self,ids):return ['<script>bad()</script>','assistant','tool']


def fixture(root,count=5):
    root.mkdir(exist_ok=True);samples=[];conversations=[]
    tokenizer=FixtureTokenizer()
    for index in range(count):
        conversation=dict(instance_id=f'fixture-{index}',messages=[dict(role='user',content='<script>bad()</script>')]+[row for turn in range(2) for row in (dict(role='assistant',tool_call=dict(id=str(turn),name='fixture',arguments={})),dict(role='tool',content='fixture',tool_call_id=str(turn)))],tools=[])
        sample=dict(instance_id=conversation['instance_id'],**tokenizer([],[]),labels=[-100,2,-100],trajectory_sha256=hashlib.sha256(canonical(conversation).encode()).hexdigest())
        samples.append(sample);conversations.append(conversation)
    (root/'samples.jsonl').write_text(''.join(canonical(row)+'\n' for row in samples))
    (root/'conversations.jsonl').write_text(''.join(canonical(row)+'\n' for row in conversations))
    manifest=dict(kind='sft_dataset_not_training_result',source_kind='software-fixture',accepted=[dict(instance_id=row['instance_id']) for row in samples],samples_sha256=digest(root/'samples.jsonl'),tokenizer={'fixture':True},mask_tokenizer=tokenizer.mask_manifest(),manual_alignment_review_status='pending')
    (root/'manifest.json').write_text(canonical(manifest)+'\n')
    return tokenizer


class MaskReviewTests(unittest.TestCase):
    def test_packet_is_escaped_hash_bound_and_pending_without_mutating_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=fixture(root/'data');before=digest(root/'data/manifest.json')
            report=prepare(root/'data',root/'review',tokenizer,tokenizer_manifest={'fixture':True})
            self.assertEqual(report['status'],'pending');self.assertEqual(report['source_scope'],'software-fixture')
            self.assertEqual(len(report['sample_ids']),5)
            page=(root/'review/review.html').read_text()
            self.assertNotIn('<script>bad()',page);self.assertIn('&lt;script&gt;bad()',page)
            rows=[json.loads(line) for line in (root/'review/tokens.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),15);self.assertEqual(rows[0]['token_id'],1);self.assertEqual(rows[0]['label'],-100)
            for name,value in report['files'].items():self.assertEqual(digest(root/'review'/name),value)
            checklist=json.loads((root/'review/human_checklist.json').read_text())
            self.assertIsNone(checklist['reviewer']);self.assertTrue(all(row['assistant_spans_correct'] is None for row in checklist['samples']))
            self.assertEqual(checklist['packet_sha256'],digest(root/'review/packet.json'))
            self.assertEqual(digest(root/'data/manifest.json'),before)
            with self.assertRaises(FileExistsError):prepare(root/'data',root/'review',tokenizer,tokenizer_manifest={'fixture':True})

    def test_insufficient_multiturn_and_tokenizer_mismatch_reject_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=fixture(root/'data',4)
            with self.assertRaisesRegex(ValueError,'five multi-turn'):prepare(root/'data',root/'review',tokenizer,tokenizer_manifest={'fixture':True})
            self.assertFalse((root/'review').exists())
            with self.assertRaisesRegex(ValueError,'tokenizer differs'):prepare(root/'data',root/'review',tokenizer,tokenizer_manifest={'fixture':False})

    def test_changed_conversation_ids_and_native_replay_are_rejected(self):
        for fault in ('hash','conversation','native','labels'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);tokenizer=fixture(root/'data')
                if fault=='conversation':
                    path=root/'data/conversations.jsonl';path.write_text(path.read_text().replace('bad()','changed()'))
                elif fault=='native':
                    class Changed(FixtureTokenizer):
                        def __call__(self,*args):return dict(input_ids=[4,2,3],assistant_mask=[0,1,0],token_roles=['user','assistant','tool'])
                    tokenizer=Changed()
                else:
                    path=root/'data/samples.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]['labels'][0]=1
                    path.write_text(''.join(canonical(row)+'\n' for row in rows))
                    if fault=='labels':
                        path=root/'data/manifest.json';manifest=json.loads(path.read_text());manifest['samples_sha256']=digest(root/'data/samples.jsonl');path.write_text(canonical(manifest)+'\n')
                with self.assertRaises(ValueError):prepare(root/'data',root/'review',tokenizer,tokenizer_manifest={'fixture':True})
                self.assertFalse((root/'review').exists())

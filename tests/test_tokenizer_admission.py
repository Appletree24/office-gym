"""Native-template protocol fixtures, not actual tokenizer/model evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from officegym.env.world import canonical
from officegym.harness.tokenizer import PromptTokenizer
from officegym.training.tokenizer import SFTTokenizer
from officegym.training.tokenizer_admission import validate_runtime


class NativeFixture:
    chat_template='AUTHORED SOFTWARE TEMPLATE FIXTURE'
    drift=False
    def get_vocab(self):return {'fixture-'+str(index):index for index in range(10)}
    def apply_chat_template(self,messages,**kwargs):
        ids=list(range(1,len(messages)+1))
        if self.drift:ids=[index+1 for index in ids]
        mask=[int(message['role']=='assistant') for message in messages]
        return dict(input_ids=ids,assistant_masks=mask) if kwargs.get('return_dict') else ids


def materialize(root,tokenizer):
    conversation=dict(instance_id='fixture',messages=[dict(role='user',content='fixture'),dict(role='assistant',tool_call=dict(id='1',name='fixture',arguments={})),dict(role='tool',content='{}',tool_call_id='1'),dict(role='assistant',tool_call=dict(id='2',name='fixture',arguments={})),dict(role='tool',content='{}',tool_call_id='2')],tools=[])
    native=SFTTokenizer(tokenizer);encoded=native(conversation['messages'],[])
    rows=[dict(instance_id='fixture',**encoded,labels=[token if mask else -100 for token,mask in zip(encoded['input_ids'],encoded['assistant_mask'])],trajectory_sha256=hashlib.sha256(canonical(conversation).encode()).hexdigest())]
    (root/'samples.jsonl').write_text(canonical(rows[0])+'\n');(root/'conversations.jsonl').write_text(canonical(conversation)+'\n')
    manifest=dict(samples_sha256=hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest(),accepted=[{'fixture':True}],tokenizer=PromptTokenizer(tokenizer,model_revision='locked-tokenizer').manifest(),mask_tokenizer=native.mask_manifest())
    return rows,manifest


class TokenizerAdmissionTests(unittest.TestCase):
    def test_explicit_tokenizer_revision_separate_from_merged_model_and_exact_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=NativeFixture();rows,manifest=materialize(root,tokenizer)
            plan=dict(model_revision='merged-model-version',tokenizer_revision='locked-tokenizer')
            report=validate_runtime(plan,manifest,root,rows,tokenizer,deadline=time.monotonic()+5)
            self.assertEqual(report['tokens'],5);self.assertEqual(report['causally_supervised_tokens'],2);self.assertTrue(report['all_native_sequences_equal'])
            with self.assertRaisesRegex(ValueError,'revision/vocabulary'):validate_runtime(dict(model_revision='different'),manifest,root,rows,tokenizer,deadline=time.monotonic()+5)
            with self.assertRaises(TimeoutError):validate_runtime(plan,manifest,root,rows,tokenizer,deadline=time.monotonic()-1)

    def test_same_manifest_with_different_native_encoding_or_changed_sources_rejected(self):
        for fault in ('native','sample','conversation','vocabulary','template'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);tokenizer=NativeFixture();rows,manifest=materialize(root,tokenizer)
                if fault=='native':tokenizer.drift=True
                elif fault=='sample':(root/'samples.jsonl').write_text('{}\n')
                elif fault=='conversation':
                    path=root/'conversations.jsonl';path.write_text(path.read_text().replace('fixture"','changed"',1))
                elif fault=='vocabulary':tokenizer.get_vocab=lambda:{'different':0}
                else:tokenizer.chat_template+=' changed'
                with self.assertRaises(ValueError):validate_runtime(dict(model_revision='locked-tokenizer'),manifest,root,rows,tokenizer,deadline=time.monotonic()+5)

    def test_all_samples_are_replayed_including_input_tokens_after_the_first_five(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=NativeFixture();original,manifest=materialize(root,tokenizer)
            conversation=json.loads((root/'conversations.jsonl').read_text());rows=[];conversations=[]
            for index in range(6):
                source=copy.deepcopy(conversation);source['instance_id']=f'fixture-{index}'
                row=copy.deepcopy(original[0]);row['instance_id']=source['instance_id'];row['trajectory_sha256']=hashlib.sha256(canonical(source).encode()).hexdigest()
                rows.append(row);conversations.append(source)
            rows[-1]['input_ids'][0]=9  # Changed user input, still label -100 and valid assistant mask.
            (root/'samples.jsonl').write_text(''.join(canonical(row)+'\n' for row in rows))
            (root/'conversations.jsonl').write_text(''.join(canonical(row)+'\n' for row in conversations))
            manifest.update(accepted=[{}]*6,samples_sha256=hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError,'native runtime token/role/mask'):validate_runtime(dict(model_revision='locked-tokenizer'),manifest,root,rows,tokenizer,deadline=time.monotonic()+5)

"""Adversarial structural review gate tests; all decisions are authored fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import canonical
from officegym.training.mask_review import prepare,digest
from officegym.training.review_admission import validate_review,register,require_review,CHECKS
from test_mask_review import fixture
from review_fixtures import reviewed_fixture


class ReviewAdmissionTests(unittest.TestCase):
    def test_pending_cannot_register_and_passed_field_alone_cannot_admit(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data';tokenizer=fixture(data);review=Path(directory)/'review'
            prepare(data,review,tokenizer,tokenizer_manifest={'fixture':True})
            before=digest(data/'manifest.json')
            with self.assertRaisesRegex(ValueError,'completed'):register(data,review/'packet.json',review/'human_checklist.json')
            self.assertEqual(digest(data/'manifest.json'),before)
            manifest=json.loads((data/'manifest.json').read_text());manifest['manual_alignment_review_status']='passed'
            with self.assertRaisesRegex(ValueError,'hash-bound'):require_review(data,manifest)

    def test_registration_is_bound_and_does_not_create_manifest_hash_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data';review=reviewed_fixture(data)
            manifest=json.loads((data/'manifest.json').read_text())
            self.assertEqual(manifest['manual_alignment_review_status'],'passed')
            self.assertEqual(len(require_review(data,manifest)['samples']),5)
            self.assertEqual(register(data,review/'packet.json',review/'human_checklist.json'),manifest['manual_alignment_review'])
            for name in ('review.html','tokens.jsonl','source_manifest.json','packet.json','human_checklist.json'):
                path=review/name;before=path.read_bytes();path.write_bytes(before+b' ')
                with self.subTest(name=name),self.assertRaises((ValueError,json.JSONDecodeError)):require_review(data,manifest)
                path.write_bytes(before)
            for name in ('samples.jsonl','conversations.jsonl'):
                path=data/name;before=path.read_bytes();path.write_bytes(before+b' ')
                with self.subTest(name=name),self.assertRaises(ValueError):require_review(data,manifest)
                path.write_bytes(before)
            changed=dict(manifest,max_length=777)
            (data/'manifest.json').write_text(canonical(changed)+'\n')
            with self.assertRaisesRegex(ValueError,'manifest changed'):require_review(data,changed)

    def test_each_decision_named_reviewer_and_aware_timestamp_required(self):
        changes=[lambda c:c.update(reviewer=''),lambda c:c.update(reviewed_at_utc='2000-01-01'),lambda c:c['samples'].pop(),lambda c:c['samples'][0].update(notes=''),lambda c:c.update(packet_sha256='0'*64)]
        changes += [lambda c,key=key:c['samples'][0].update({key:1}) for key in CHECKS]
        changes += [lambda c:c['samples'].reverse()]
        for change in changes:
            with tempfile.TemporaryDirectory() as directory:
                data=Path(directory)/'data';review=reviewed_fixture(data);path=review/'human_checklist.json';checklist=json.loads(path.read_text());change(checklist);path.write_text(canonical(checklist)+'\n')
                with self.assertRaises(ValueError):validate_review(data,review/'packet.json',path)

    def test_sft_entrypoint_rejects_status_only_after_other_preflight_checks(self):
        from officegym.training.sft import validate_plan
        from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
        from officegym.harness.versions import HARNESS_VERSION
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data'
            reviewed_fixture(data,dict(harness_version=HARNESS_VERSION,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION))
            plan=dict(model_path=directory,model_revision='authored-fixture',max_steps=1,max_wall_seconds=60,checkpoint_interval_steps=1,locked_packages={name:'fixture' for name in ('torch','transformers','trl','peft','datasets')},max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=.0001,lora={})
            self.assertEqual(validate_plan(plan,data)['manual_alignment_review_status'],'passed')
            manifest=json.loads((data/'manifest.json').read_text());del manifest['manual_alignment_review']
            (data/'manifest.json').write_text(canonical(manifest)+'\n')
            with self.assertRaisesRegex(ValueError,'hash-bound human'):validate_plan(plan,data)

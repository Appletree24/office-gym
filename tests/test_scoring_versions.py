import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tasks.validate import validate_dataset
from officegym.training.teacher import extract_trajectory
from officegym.training.data import RejectedTrajectory


class ScoringVersionTests(unittest.TestCase):
    def test_rehashed_old_reference_grade_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'data'
            manifest=generate(root,templates=['T01'],count_per_template=1)
            entry=manifest['instances'][0]
            grade_path=root/entry['path']/'reference_grade.json'
            grade=json.loads(grade_path.read_text())
            grade['eval_grader_version']='old'
            grade_path.write_text(json.dumps(grade))
            entry['files_sha256']['reference_grade.json']=hashlib.sha256(grade_path.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'stale or missing scoring'):
                validate_dataset(root/'manifest.json')

    def test_successful_model_marker_cannot_bypass_version_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'result.json').write_text(json.dumps(dict(split='train',policy_kind='inference_service',grade={'full_success':True})))
            with self.assertRaisesRegex(RejectedTrajectory,'stale_scoring_version'):
                extract_trajectory(root)

    def test_optimizer_preflight_rejects_stale_reviewed_dataset(self):
        from officegym.training.sft import validate_plan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=dict(model_path=directory,model_revision='locked',max_steps=1,checkpoint_interval_steps=1,max_wall_seconds=60,
                      locked_packages={name:'locked' for name in ('torch','transformers','trl','peft','datasets')},
                      max_sequence_tokens=8192,micro_batch_size=1,gradient_accumulation_steps=1,learning_rate=0.0001,lora={})
            (root/'manifest.json').write_text(json.dumps(dict(manual_alignment_review_status='passed',accepted=[{}])))
            with self.assertRaisesRegex(ValueError,'stale or missing scoring'):
                validate_plan(plan,root)

"""Native checkpoint protocol byte fixtures; no FSDP or model training occurs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from officegym.training.evidence import TrainingEvidence
from officegym.training.verl_checkpoint import validate_profile,record_saved_checkpoint,observed_trainer_class,observed_tracking_class,extract_method
from lineage_fixtures import checkpoint_profile


class VerlCheckpointTests(unittest.TestCase):
    def saved(self,root):
        config=checkpoint_profile(root);folder=root/'checkpoints/global_step_2';actor=folder/'actor';(actor/'huggingface').mkdir(parents=True)
        (actor/'fsdp_config.json').write_text('{"FSDP_version":2,"world_size":2}')
        for rank in (0,1):
            for kind in ('model','optim','extra_state'):(actor/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'authored shard, not model')
        (actor/'huggingface/config.json').write_text('{}');(actor/'huggingface/tokenizer_config.json').write_text('{}');(folder/'data.pt').write_bytes(b'authored dataloader state')
        (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('2')
        return config,folder,actor

    def test_completed_all_rank_save_binds_actor_and_resume_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,folder,actor=self.saved(root)
            record=record_saved_checkpoint(config,2)
            events=[json.loads(row) for row in (root/'training_events.jsonl').read_text().splitlines()]
            self.assertEqual([r['kind'] for r in events],['resume_checkpoint','training_checkpoint'])
            self.assertIn('data.pt',events[0]['payload']['files_sha256'])
            self.assertEqual(record['path'],str(actor))
            self.assertEqual(events[-1]['optimizer_step'],2)
            self.assertIn('extra_state_world_size_2_rank_1.pt',record['files_sha256'])

    def test_partial_or_inconsistent_save_emits_no_checkpoint_event(self):
        for missing in ('model_world_size_2_rank_1.pt','optim_world_size_2_rank_0.pt','extra_state_world_size_2_rank_1.pt','huggingface/tokenizer_config.json'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);config,folder,actor=self.saved(root);(actor/missing).unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):record_saved_checkpoint(config,2)
                self.assertFalse((root/'training_events.jsonl').exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,folder,actor=self.saved(root)
            (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('1')
            with self.assertRaisesRegex(ValueError,'completion marker'):record_saved_checkpoint(config,2)
            self.assertFalse((root/'training_events.jsonl').exists())

    def test_unsafe_checkpoint_profiles_rejected_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            config=checkpoint_profile(directory);validate_profile(config)
            for changes in ({'use_v1':True},{'save_freq':0},{'save_freq':True},{'max_actor_ckpt_to_keep':1},{'resume_mode':'auto'},{'default_hdfs_dir':'remote'},{'default_local_dir':'/outside/run'}):
                altered=copy.deepcopy(config);altered['trainer'].update(changes)
                with self.subTest(changes=changes),self.assertRaises(ValueError):validate_profile(altered)
            for changes in ({'async_save':True},{'save_contents':['model']}):
                altered=copy.deepcopy(config);altered['actor_rollout_ref']['actor']['checkpoint'].update(changes)
                with self.assertRaises(ValueError):validate_profile(altered)

    def test_save_failure_is_not_logged_as_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=checkpoint_profile(root);evidence=TrainingEvidence(root)
            class Failing:
                global_steps=2
                _officegym_optimizer_steps=2
                def _save_checkpoint(self):raise RuntimeError('authored native save failure')
            observed=observed_trainer_class(Failing,config,evidence)
            with self.assertRaises(RuntimeError):observed()._save_checkpoint()
            self.assertFalse((root/'training_events.jsonl').exists())
            class Tracking:
                def log(self,data,step):return 'native return'
            tracker=observed_tracking_class(Tracking,5,evidence)()
            self.assertEqual(tracker.log({'loss':float('inf')},2),'native return')
            row=json.loads((root/'training_events.jsonl').read_text())
            self.assertEqual(row['payload']['loss'],{'non_finite':'inf'})
            with self.assertRaises(ValueError):tracker.log({},6)

    def test_method_extraction_preserves_body_and_source_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.py';path.write_text('class Controller:\n def run(self, config):\n  return native(config)\n')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            method=extract_method(path,sha,'Controller','run',{'native':lambda x:x+1})
            self.assertEqual(method(None,2),3)
            with self.assertRaisesRegex(ValueError,'source changed'):extract_method(path,'bad','Controller','run',{})

    def test_optimizer_count_is_verified_instead_of_assuming_global_step(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=checkpoint_profile(root);evidence=TrainingEvidence(root)
            class Batch:
                def __len__(self):return 4
            class Controller:
                global_steps=1
                norm=[.5]
                def _update_actor(self,batch):return SimpleNamespace(meta_info={'metrics':{'actor/grad_norm':self.norm}})
                def _save_checkpoint(self):raise AssertionError('must not save an unverified step')
            observed=observed_trainer_class(Controller,config,evidence)()
            observed._update_actor(Batch())
            self.assertEqual(observed._officegym_optimizer_steps,1)
            events=[json.loads(line) for line in (root/'training_events.jsonl').read_text().splitlines()]
            self.assertEqual([row['kind'] for row in events],['optimizer_update_attempt','optimizer_update'])
            observed.global_steps=2;observed.norm=[float('nan')]
            with self.assertRaisesRegex(ValueError,'finite gradient'):observed._update_actor(Batch())
            self.assertEqual(observed._officegym_optimizer_steps,1)
            events=[json.loads(line) for line in (root/'training_events.jsonl').read_text().splitlines()]
            self.assertEqual([row['kind'] for row in events],['optimizer_update_attempt','optimizer_update','optimizer_update_attempt'])
            with self.assertRaisesRegex(ValueError,'verified optimizer counter'):observed._save_checkpoint()

    def test_recorded_native_step_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,folder,actor=self.saved(root);record_saved_checkpoint(config,2)
            evidence=TrainingEvidence(root);weight=actor/'model_world_size_2_rank_0.pt';before=weight.read_bytes()
            class Controller:
                global_steps=2
                _officegym_optimizer_steps=2
                def _save_checkpoint(self):weight.write_bytes(b'overwritten')
            with self.assertRaisesRegex(ValueError,'refusing overwrite'):
                observed_trainer_class(Controller,config,evidence)()._save_checkpoint()
            self.assertEqual(weight.read_bytes(),before)

    def test_export_requires_complete_finite_optimizer_prefix(self):
        from officegym.training.checkpoints import verify_grpo_updates
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            rows=[dict(kind='optimizer_update',optimizer_step=step,payload=dict(native_global_step=step,contract='single-mini-batch-epoch',gradient_norms=[.5])) for step in (1,2)]
            for data in (rows[:1],list(reversed(rows)),[dict(rows[0],payload=dict(rows[0]['payload'],gradient_norms=[float('nan')])),rows[1]]):
                path.write_text(''.join(json.dumps(row)+'\n' for row in data))
                with self.assertRaises(ValueError):verify_grpo_updates(path,path.stat().st_size,2)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows));prefix=path.stat().st_size
            with path.open('a') as stream:stream.write(json.dumps(dict(kind='optimizer_update',optimizer_step=3,payload={}))+'\n')
            self.assertEqual(verify_grpo_updates(path,prefix,2),2)

    def test_formal_grpo_source_requires_the_native_code_lock(self):
        from officegym.training.checkpoints import verify_grpo_source_lock
        from officegym.training.verl_checkpoint import SOURCE_HASHES
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            good=dict(kind='backend_source_lock',optimizer_step=0,payload=SOURCE_HASHES)
            for rows in ([],[dict(good,payload={})],[good,good]):
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaises(ValueError):verify_grpo_source_lock(path,path.stat().st_size)
            path.write_text(json.dumps(good)+'\n')
            self.assertEqual(verify_grpo_source_lock(path,path.stat().st_size),SOURCE_HASHES)

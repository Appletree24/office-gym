import unittest
from officegym.training.launch import validate_launch,compatibility_hash
from officegym.training.verl_entry import bound_command,load_resolved


class LaunchTests(unittest.TestCase):
    def test_intent_config_not_accepted_as_upstream_launch(self):
        with self.assertRaisesRegex(ValueError,'missing locked launch fields'):
            validate_launch({'algorithm':'grpo'})

    def test_timeout_cleans_process_group(self):
        import os
        import sys
        import tempfile
        from pathlib import Path
        from officegym.training.launch import execute_bounded
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'out').open('w') as stdout, (root/'err').open('w') as stderr:
                result = execute_bounded([sys.executable,'-c','import time; time.sleep(30)'],stdout,stderr,.1)
            self.assertTrue(result['wall_budget_exhausted'])
            self.assertTrue(result['process_group_cleanup'])
            self.assertNotEqual(result['exit_code'],0)

    def test_actual_upstream_group_size_is_checked(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from officegym.training.launch import VERL_COMMIT
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)/'resolved.json'
            upstream.write_text(json.dumps({'algorithm':{'adv_estimator':'grpo'},'actor_rollout_ref':{'rollout':{'n':4}},'trainer':{'total_training_steps':5}}))
            config = dict(verl_commit=VERL_COMMIT,verl_checkout=directory,locked_packages={},resolved_upstream_config=str(upstream),resolved_upstream_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),command=['python','verified_entrypoint.py'],max_updates=5,max_wall_seconds=60,max_rollout_tokens=1000,starting_checkpoint_hash='sft_hash',reference_checkpoint_hash='sft_hash',group_size=4,rollout_contract=dict(token_ledger_path=str(Path(directory)/'tokens.sqlite'),artifact_root=directory,global_rollout_token_limit=1000,max_generation_tokens=100,max_context_tokens=8192,max_tool_calls=24,actor_version='sft_hash'))
            from lineage_fixtures import exported_sft_fixture
            checkpoint,export,record=exported_sft_fixture(Path(directory)/'sft_source')
            config.update(starting_checkpoint_path=str(checkpoint),starting_export=str(export),starting_checkpoint_hash=record['checkpoint_sha256'],reference_checkpoint_hash=record['checkpoint_sha256'])
            config['rollout_contract']['actor_version']=record['checkpoint_sha256']
            from officegym.tasks.generate import generate
            from officegym.training.prepare_rl import prepare
            tasks=Path(directory)/'tasks'
            generate(tasks,templates=['T01'],count_per_template=1,split='train')
            prepared=prepare(tasks/'manifest.json',config['rollout_contract'],Path(directory)/'prepared')
            train=Path(prepared['train_file'])
            original_train=train.read_text()
            plugins=prepared['worker_plugins']
            preparation=Path(directory)/'prepared'/'preparation.json'
            config['prepared_rollouts'] = str(preparation)
            resolved = json.loads(upstream.read_text())
            from lineage_fixtures import checkpoint_profile
            profile=checkpoint_profile(Path(directory)/'run')
            profile['trainer']['officegym_verl_checkout']=directory
            resolved['trainer'].update(profile['trainer'])
            resolved['trainer']['officegym_starting_actor_hash']=config['starting_checkpoint_hash']
            resolved['actor_rollout_ref']['actor']=profile['actor_rollout_ref']['actor']
            resolved['actor_rollout_ref']['model']={'path':str(checkpoint),'lora_rank':16,'lora_alpha':32}
            resolved['actor_rollout_ref']['actor'].update(use_kl_loss=True,kl_loss_coef=.001)
            resolved['actor_rollout_ref']['rollout'].update(name='vllm',load_format='auto')
            resolved['data'] = dict(train_files=[str(train)],train_batch_size=2)
            resolved['actor_rollout_ref']['rollout'].update(agent={'agent_loop_config_path':plugins['agent_loop']['path']},multi_turn={'tool_config_path':plugins['tools']['path']})
            upstream.write_text(json.dumps(resolved))
            config['resolved_upstream_sha256'] = hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command'] = bound_command(upstream,config['resolved_upstream_sha256'])
            self.assertEqual(validate_launch(config,smoke=True),config['command'])
            resolved['actor_rollout_ref']['actor']['use_kl_loss']=False
            upstream.write_text(json.dumps(resolved));config['resolved_upstream_sha256']=hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command']=bound_command(upstream,config['resolved_upstream_sha256'])
            with self.assertRaisesRegex(ValueError,'reference KL'):
                validate_launch(config,smoke=True)
            resolved['actor_rollout_ref']['actor']['use_kl_loss']=True
            resolved['actor_rollout_ref']['model']['path']='/another/model'
            upstream.write_text(json.dumps(resolved));config['resolved_upstream_sha256']=hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command']=bound_command(upstream,config['resolved_upstream_sha256'])
            with self.assertRaisesRegex(ValueError,'algorithm/group/update'):
                validate_launch(config,smoke=True)
            resolved['actor_rollout_ref']['model']['path']=str(checkpoint)
            upstream.write_text(json.dumps(resolved));config['resolved_upstream_sha256']=hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command']=bound_command(upstream,config['resolved_upstream_sha256'])
            config['command'] = ['python','another_entrypoint.py']
            with self.assertRaisesRegex(ValueError,'command must execute'):
                validate_launch(config,smoke=True)
            config['command'] = bound_command(upstream,config['resolved_upstream_sha256'])
            from officegym.training.launch import SMOKE_CHECKS
            report = Path(directory)/'smoke.json'
            report.write_text(json.dumps(dict(passed=True,passed_checks=sorted(SMOKE_CHECKS),compatibility_sha256=compatibility_hash(resolved))))
            config['smoke_report'] = str(report)
            config['max_updates'] = 100
            resolved['trainer']['total_training_steps'] = 100
            upstream.write_text(json.dumps(resolved))
            config['resolved_upstream_sha256'] = hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command'] = bound_command(upstream,config['resolved_upstream_sha256'])
            self.assertEqual(validate_launch(config,smoke=False),config['command'])
            config['max_updates'] = 5
            resolved['trainer']['total_training_steps'] = 5
            upstream.write_text(json.dumps(resolved))
            config['resolved_upstream_sha256'] = hashlib.sha256(upstream.read_bytes()).hexdigest()
            config['command'] = bound_command(upstream,config['resolved_upstream_sha256'])
            train.write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError,'data hash mismatch'):
                validate_launch(config,smoke=True)
            train.write_text(original_train)
            resolved['data']['train_files'] = ['/another/train.jsonl']
            upstream.write_text(json.dumps(resolved))
            config['resolved_upstream_sha256'] = hashlib.sha256(upstream.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'train files differ'):
                validate_launch(config,smoke=True)
            config['group_size'] = 2
            with self.assertRaisesRegex(ValueError,'algorithm/group/update'):
                validate_launch(config,smoke=True)

    def test_smoke_compatibility_allows_only_update_count_change(self):
        smoke={'trainer':{'total_training_steps':5},'algorithm':{'adv_estimator':'grpo'},'actor':{'learning_rate':0.000001}}
        training={**smoke,'trainer':{'total_training_steps':100}}
        self.assertEqual(compatibility_hash(smoke),compatibility_hash(training))
        changed={**training,'actor':{'learning_rate':0.00001}}
        self.assertNotEqual(compatibility_hash(smoke),compatibility_hash(changed))

    def test_bound_loader_rejects_changed_bytes_and_unresolved_config(self):
        import tempfile,json,hashlib
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            path.write_text(json.dumps({'model':'${unresolved}'}))
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'resolved interpolations'):
                load_resolved(path,checksum)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed before execution'):
                load_resolved(path,checksum)

    def test_explicit_smoke_update_request_cannot_override_locked_launch(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from officegym.training.launch import launch
        for updates in (0,6,True,float('nan'),'5'):
            with self.subTest(updates=updates),self.assertRaisesRegex(ValueError,'one to five'):
                launch('missing','missing',True,expected_updates=updates)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'launch.json';config.write_text(json.dumps({'max_updates':4}))
            with patch('officegym.training.launch.validate_launch') as validate:
                with self.assertRaisesRegex(ValueError,'exactly match'):launch(config,root/'run',True,expected_updates=5)
                validate.assert_not_called();self.assertFalse((root/'run').exists())
            with self.assertRaisesRegex(ValueError,'one to five'):launch(config,root/'run',False,expected_updates=4)

    def test_cli_process_success_cannot_claim_verified_semantic_smoke(self):
        import contextlib,io
        from unittest.mock import patch
        from officegym.cli import main
        base=dict(exit_code=0,wall_budget_exhausted=False,process_tree_cleanup=True,semantic_smoke_verification='pending')
        for changes,exit_code in (({},2),({'exit_code':1},1),({'process_tree_cleanup':False},1),({'wall_budget_exhausted':True},1),({'semantic_smoke_verification':'verified'},0)):
            with self.subTest(changes=changes),patch('sys.argv',['officegym','grpo-smoke','--config','locked.json','--out','run','--updates','5']),patch('officegym.training.launch.launch',return_value={**base,**changes}) as launch,contextlib.redirect_stdout(io.StringIO()):
                if exit_code:
                    with self.assertRaises(SystemExit) as error:main()
                    self.assertEqual(error.exception.code,exit_code)
                else:main()
                launch.assert_called_once_with('locked.json','run',True,expected_updates=5)

import json
import tempfile
import unittest
from pathlib import Path
from officegym.training.teacher import extract_trajectory
from officegym.training.data import RejectedTrajectory


class TeacherExtractTests(unittest.TestCase):
    def test_nontraining_and_scripted_runs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'result.json').write_text(json.dumps({'split':'id_test'}))
            with self.assertRaisesRegex(RejectedTrajectory,'non_training_split'):
                extract_trajectory(root)
            (root/'result.json').write_text(json.dumps({'split':'train','policy_kind':'software_fixture'}))
            with self.assertRaisesRegex(RejectedTrajectory,'not_a_model'):
                extract_trajectory(root)

    def test_fixture_response_action_and_training_messages_must_match(self):
        # Entire episode is an authored protocol fixture, not model evidence.
        from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
        from officegym.harness.versions import HARNESS_VERSION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=dict(split='train',policy_kind='inference_service',instance_id='fixture',model_version='fixture',
                        grade=dict(full_success=True,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION),
                        termination={'state':'DECLARED_DONE'},operation_audit={'consistent':True})
            # Authored metadata fixture for parser acceptance; not an OS-isolation proof.
            from officegym.harness.isolation import ISOLATION_VERSION
            evidence=dict(version=ISOLATION_VERSION,verified=True,uid=65534,gid=65534,groups=[],no_new_privs=True,dumpable=0,cannot_regain_root=True,grader_unavailable=True,root_entries=[],inherited_descriptors=[0,1,2],parent_death_signal='SIGKILL',environment_empty=True,protected_paths_denied=['/fixture/'+str(i) for i in range(5)])
            result['policy_isolation']=evidence
            (root/'policy_isolation.json').write_text(json.dumps(evidence))
            from teacher_world_fixtures import bind_world_fixture
            bind_world_fixture(root,result)
            (root/'result.json').write_text(json.dumps(result))
            records=[dict(kind='run_header',harness_version=HARNESS_VERSION,mode='full',model_version='fixture',initial_messages=[{'role':'user','content':'fixture task'}],tool_schemas=[]),
                     dict(kind='model_response',tool='task.finish',arguments={'summary':'done'},content='The task is complete.',raw={'fixture':True}),
                     dict(kind='action_start',action_id='a',tool='task.finish',arguments={'summary':'done'}),
                     dict(kind='interaction',action_id='a',messages=[{'role':'assistant','content':'The task is complete.','tool_call':{'id':'1','name':'task.finish','arguments':{'summary':'done'}}},{'role':'tool','tool_call_id':'1','content':'{}'}])]
            def save():
                (root/'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            save()
            source=extract_trajectory(root)
            self.assertEqual(source['conversation'][1]['content'],'The task is complete.')
            records.insert(1,dict(kind='context_selection',compression_triggered=True,summary_output_tokens=0))
            save()
            with self.assertRaisesRegex(RejectedTrajectory,'compressed_or_truncated'):
                extract_trajectory(root)
            records.pop(1)
            records[3]['messages'][0]['tool_call']['arguments']={'summary':'changed'}
            save()
            with self.assertRaisesRegex(RejectedTrajectory,'interaction_differs'):
                extract_trajectory(root)
            records.insert(1,dict(kind='parse_failure',message={'role':'user','content':'retry'}))
            save()
            with self.assertRaisesRegex(RejectedTrajectory,'parse_retried'):
                extract_trajectory(root)

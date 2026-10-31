import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from officegym.harness.isolation import IsolatedPolicy
from officegym.harness.policy import ChatPolicy


def fixture_transport(payload,timeout):
    if 'tools' in payload:
        message=dict(content='fixture rationale',tool_calls=[dict(function=dict(name='task.finish',arguments='{"summary":"fixture"}'))])
    else:
        message=dict(content='{"goals":[]}')
    return dict(choices=[dict(message=message)],usage=dict(prompt_tokens=5,completion_tokens=3))


@unittest.skipUnless(os.name=='posix' and os.geteuid()==0,'requires Linux chroot/setuid launcher privileges')
class PolicyIsolationTests(unittest.TestCase):
    def test_private_paths_privilege_descriptors_and_rpc(self):
        with tempfile.TemporaryDirectory() as directory:
            secret=Path(directory)/'private_spec.json'
            secret.write_text('PRIVATE_GROUND_TRUTH')
            # Keep an actual private fd open in the trusted parent during exec.
            with secret.open() as inherited:
                os.set_inheritable(inherited.fileno(),True)
                policy=ChatPolicy('fixture','model','authored-fixture',transport=fixture_transport)
                worker=IsolatedPolicy(policy,protected_paths=[secret,'/proc/1/root'+str(secret),'/../../'+str(secret).lstrip('/')])
                root=worker.root
                process=worker.process
                try:
                    evidence=worker.evidence
                    self.assertTrue(evidence['verified'])
                    self.assertEqual(evidence['inherited_descriptors'],[0,1,2])
                    self.assertEqual(evidence['root_entries'],[])
                    self.assertEqual(len(evidence['protected_paths_denied']),3)
                    self.assertTrue(evidence['cannot_regain_root'])
                    self.assertTrue(evidence['grader_unavailable'])
                    self.assertEqual(evidence['parent_death_signal'],'SIGKILL')
                    remaining=dict(wall_seconds=2,generated_tokens=10)
                    turn=worker.generate([dict(role='user',content='public')],[],remaining)
                    self.assertEqual(turn.tool,'task.finish')
                    self.assertEqual(turn.content,'fixture rationale')
                    self.assertEqual((turn.input_tokens,turn.output_tokens),(5,3))
                    self.assertEqual(worker.summarize([],remaining)['summary'],dict(goals=[]))
                finally:
                    worker.close()
                self.assertIsNotNone(process.poll())
                self.assertFalse(root.exists())

    def test_dead_worker_fails_and_cleanup_is_bounded(self):
        worker=IsolatedPolicy(ChatPolicy('fixture','model','authored-fixture',transport=fixture_transport))
        try:
            worker.process.kill()
            worker.process.wait(timeout=2)
            with self.assertRaises((RuntimeError,BrokenPipeError)):
                worker.generate([],[],dict(wall_seconds=.1,generated_tokens=5))
        finally:
            worker.close()

    def test_privilege_failure_has_no_unisolated_fallback(self):
        with patch('officegym.harness.isolation.os.geteuid',return_value=65534):
            with self.assertRaisesRegex(RuntimeError,'no unsafe fallback'):
                IsolatedPolicy(ChatPolicy('fixture','model','authored-fixture',transport=fixture_transport))

    def test_episode_uses_worker_and_restores_summary_policy(self):
        from officegym.tasks.generate import generate
        from officegym.eval.episode import run_episode
        from officegym.harness.context import FullContext
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            policy=ChatPolicy('fixture','model','authored-fixture',transport=fixture_transport)
            context=FullContext(lambda *args:1,100,policy,trigger_tokens=20,reserved_output_tokens=10)
            result=run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',policy,context_manager=context)
            self.assertEqual(result['termination']['state'],'DECLARED_DONE')
            self.assertIsNone(result['infrastructure_error'])
            self.assertFalse(result['grade']['full_success'])
            self.assertIs(context.policy,policy)
            evidence=json.loads((root/'run/policy_isolation.json').read_text())
            self.assertTrue(evidence['verified'])
            self.assertEqual(len(evidence['protected_paths_denied']),5)
            from officegym.harness.isolation import require_isolation
            self.assertTrue(require_isolation(root/'run',result))
            self.assertEqual(result['policy_kind'],'software_fixture')
            evidence['uid']=0
            result['policy_isolation']=evidence
            (root/'run/policy_isolation.json').write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'constraints'):
                require_isolation(root/'run',result)

    def test_startup_failure_is_retained_and_export_gate_rejects_tampering(self):
        from officegym.tasks.generate import generate
        from officegym.eval.episode import run_episode
        from officegym.harness.isolation import require_isolation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root/'tasks',templates=['T01'],count_per_template=1)
            policy=ChatPolicy('fixture','model','authored-fixture',transport=fixture_transport)
            with patch('officegym.harness.isolation.os.geteuid',return_value=65534):
                result=run_episode(root/'tasks'/manifest['instances'][0]['path'],root/'run',policy)
            self.assertEqual(result['termination']['state'],'INFRA_ERROR')
            self.assertFalse(result['grade']['full_success'])
            self.assertFalse(require_isolation(root/'run',result,allow_startup_failure=True))
            with self.assertRaisesRegex(ValueError,'not verified'):
                require_isolation(root/'run',result)
            evidence=json.loads((root/'run/policy_isolation.json').read_text())
            evidence['verified']=True
            (root/'run/policy_isolation.json').write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'differs'):
                require_isolation(root/'run',result,allow_startup_failure=True)

    def test_default_http_transport_works_through_parent_broker(self):
        from http.server import BaseHTTPRequestHandler,HTTPServer
        from threading import Thread
        observed=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append(payload)
                raw=json.dumps(fixture_transport(payload,1)).encode()
                self.send_response(200)
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self,*args):
                pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True)
        thread.start()
        worker=None
        try:
            policy=ChatPolicy('http://127.0.0.1:'+str(server.server_port)+'/v1/chat/completions','fixture-model','fixture-server')
            worker=IsolatedPolicy(policy)
            self.assertEqual(worker.generate([],[],dict(wall_seconds=2,generated_tokens=10)).tool,'task.finish')
            self.assertEqual(len(observed),1)
            self.assertEqual(observed[0]['model'],'fixture-model')
        finally:
            if worker:
                worker.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

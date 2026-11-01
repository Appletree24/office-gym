"""Software-only upstream and HTTP budget tests; no teacher/API measurement."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from officegym.training.teacher_gateway import CostLedger,Gateway,validate


def config(root,**changes):
    value=dict(endpoint='https://provider.invalid/v1/chat/completions',model='fixture-model',model_version='fixture-version',api_key_env='TEST_TEACHER_UPSTREAM_KEY',gateway_key_env='TEST_TEACHER_LOCAL_KEY',ledger_path=str(Path(root)/'cost.sqlite'),pricing=dict(max_cost_usd='.01',input_usd_per_million='1',output_usd_per_million='2',usage_contract='total_billable_prompt_and_completion_tokens',source='authored-test-rates-not-commercial',as_of_utc='2000-01-01T00:00:00Z'),max_requests=5,max_input_tokens=1000,max_output_tokens=10,max_wall_seconds=60,request_timeout_seconds=1)
    value.update(changes);return value


def payload():return dict(model='fixture-model',messages=[dict(role='user',content='hello')],max_tokens=10,temperature=0,stream=False)


def response():return dict(model='fixture-version',usage=dict(prompt_tokens=4,completion_tokens=2),choices=[dict(message=dict(content='fixture'))])


class TeacherGatewayTests(unittest.TestCase):
    def test_decimal_policy_and_fixed_identity_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);validate(value)
            for change in [dict(endpoint='http://provider.invalid'),dict(endpoint='https://user:secret@provider.invalid'),dict(model='SET_BY_OPERATOR'),dict(max_requests=True),dict(request_timeout_seconds=61),dict(gateway_key_env='TEST_TEACHER_UPSTREAM_KEY')]:
                with self.subTest(change=change),self.assertRaises(ValueError):validate(config(directory,**change))
            for price in ['NaN','-1',.01]:
                value=config(directory);value['pricing']['max_cost_usd']=price
                with self.subTest(price=price),self.assertRaises(ValueError):validate(value)

    def test_reservations_release_only_confirmed_unused_cost_and_restart_preserves_spend(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);value['pricing']['max_cost_usd']='.00102'
            ledger=CostLedger(value);identity=ledger.reserve('0'*64,10)
            self.assertEqual(ledger.summary()['charged_upper_nanousd'],1020000)
            with self.assertRaises(ValueError):ledger.reserve('1'*64,10)
            ledger.observe(identity,dict(prompt_tokens=4,completion_tokens=2));ledger.close()
            ledger=CostLedger(value);self.assertEqual(ledger.summary()['known_token_price_nanousd'],8000)
            self.assertEqual(ledger.summary()['unresolved_requests'],0)
            with self.assertRaises(ValueError):ledger.observe(identity,dict(prompt_tokens=4,completion_tokens=2))
            ledger.close()
            changed=config(directory);changed['pricing']['max_cost_usd']='1'
            with self.assertRaisesRegex(ValueError,'cannot change'):CostLedger(changed)

    def test_parallel_callers_cannot_exceed_reserved_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory,max_requests=10);value['pricing']['max_cost_usd']='.00102'
            ledger=CostLedger(value);ledger.close()
            def reserve(index):
                ledger=CostLedger(value)
                try:
                    try:ledger.reserve(f'{index:064x}',10);return True
                    except ValueError:return False
                finally:ledger.close()
            with ThreadPoolExecutor(max_workers=8) as workers:results=list(workers.map(reserve,range(8)))
            self.assertEqual(sum(results),1)
            ledger=CostLedger(value);self.assertEqual(ledger.summary()['unresolved_requests'],1);ledger.close()

    def test_provider_bound_breach_is_recorded_and_stops_future_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);ledger=CostLedger(value);identity=ledger.reserve('a'*64,10)
            with self.assertRaisesRegex(ValueError,'provider exceeded'):ledger.observe(identity,dict(prompt_tokens=4,completion_tokens=11))
            self.assertTrue(ledger.summary()['provider_contract_breach'])
            self.assertEqual(ledger.summary()['known_token_price_nanousd'],26000)
            with self.assertRaises(ValueError):ledger.reserve('b'*64,10)
            ledger.close()

    @patch.dict(os.environ,{'TEST_TEACHER_LOCAL_KEY':'local-fixture-secret'})
    def test_timeout_keeps_full_reservation_and_rejects_unbounded_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory)
            def timeout(data,seconds):raise TimeoutError('authored upstream uncertainty')
            gateway=Gateway(value,timeout)
            for change in [dict(n=20),dict(max_completion_tokens=100),dict(stream=True),dict(model='substituted'),dict(tools=[{'type':'web_search'}]),dict(messages=[dict(role='user',content=[{'type':'image_url'}])])]:
                with self.subTest(change=change),self.assertRaises(ValueError):gateway.request({**payload(),**change})
            with self.assertRaises(TimeoutError):gateway.request(payload())
            ledger=CostLedger(value);summary=ledger.summary();ledger.close()
            self.assertEqual(summary['requests'],1);self.assertEqual(summary['charged_upper_nanousd'],1020000)
            self.assertEqual(summary['known_token_price_nanousd'],0);self.assertEqual(summary['unresolved_requests'],1)
            gateway.deadline=0
            with self.assertRaisesRegex(ValueError,'lifetime'):gateway.request(payload())

    @patch.dict(os.environ,{'TEST_TEACHER_LOCAL_KEY':'local-fixture-secret'})
    def test_wrong_provider_identity_stops_calls_after_retaining_measured_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);calls=[]
            def wrong(data,seconds):calls.append(1);return {**response(),'model':'another-model'}
            gateway=Gateway(value,wrong)
            with self.assertRaisesRegex(ValueError,'model identity'):gateway.request(payload())
            with self.assertRaises(ValueError):gateway.request(payload())
            self.assertEqual(len(calls),1)
            ledger=CostLedger(value);self.assertEqual(ledger.summary()['known_token_price_nanousd'],8000);self.assertTrue(ledger.summary()['provider_contract_breach']);ledger.close()

    @patch.dict(os.environ,{'TEST_TEACHER_LOCAL_KEY':'local-fixture-secret'})
    def test_real_loopback_auth_and_fixture_cannot_become_model_trajectory(self):
        from officegym.harness.policy import ChatPolicy
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);calls=[]
            def fixture(data,seconds):calls.append(1);return response()
            gateway=Gateway(value,fixture);server=gateway.server();thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
            try:
                with self.assertRaises(HTTPError) as error:urlopen(Request(endpoint,data=json.dumps(payload()).encode()),timeout=2)
                self.assertEqual(error.exception.code,401);self.assertEqual(calls,[])
                request=Request(endpoint,data=json.dumps(payload()).encode(),headers={'Authorization':'Bearer local-fixture-secret','Content-Type':'application/json'})
                with urlopen(request,timeout=2) as result:body=json.load(result)
                self.assertEqual(body['_officegym_gateway']['source_kind'],'software_fixture')
                self.assertNotIn('local-fixture-secret',json.dumps(body))
                policy=ChatPolicy(endpoint,'fixture-model','fixture-version',api_key_env='TEST_TEACHER_LOCAL_KEY',max_output_tokens=10)
                with self.assertRaisesRegex(ValueError,'software gateway fixture'):policy._request(payload()['messages'],dict(generated_tokens=10,wall_seconds=2))
                with urlopen(Request(endpoint.rsplit('/v1/',1)[0]+'/budget',headers={'Authorization':'Bearer local-fixture-secret'}),timeout=2) as result:summary=json.load(result)
                self.assertEqual(summary['requests'],2);self.assertEqual(len(calls),2)
            finally:server.shutdown();server.server_close();thread.join(2)

    def test_gateway_identity_cannot_be_mislabelled_by_chat_adapter(self):
        from officegym.harness.policy import ChatPolicy
        body={**response(),'_officegym_gateway':dict(source_kind='configured_upstream',model_version='another-version')}
        policy=ChatPolicy('https://unused.invalid','fixture-model','fixture-version',transport=lambda data,seconds:body,max_output_tokens=10)
        with self.assertRaisesRegex(ValueError,'model version differs'):policy._request(payload()['messages'],dict(generated_tokens=10,wall_seconds=1))

    def test_upstream_redirect_does_not_forward_authorization_to_destination(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from urllib.request import build_opener
        from officegym.training.teacher_gateway import NoRedirect
        hits=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                hits.append(self.path);self.send_response(302);self.send_header('Location','/destination');self.end_headers()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            request=Request(f'http://127.0.0.1:{server.server_port}/redirect',headers={'Authorization':'Bearer authored-secret'})
            with self.assertRaisesRegex(ValueError,'redirects'):build_opener(NoRedirect()).open(request,timeout=2)
            self.assertEqual(hits,['/redirect'])
        finally:server.shutdown();server.server_close();thread.join(2)

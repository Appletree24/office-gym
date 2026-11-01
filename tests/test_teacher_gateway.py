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


class TeacherCostSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_known_and_unknown_without_keys_or_live_ledger_mutation(self):
        from officegym.training.teacher_gateway import snapshot
        import hashlib
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);ledger=CostLedger(value)
            self.assertEqual(ledger.db.execute('PRAGMA journal_mode=WAL').fetchone()[0],'wal')
            identity=ledger.reserve('0'*64,10);ledger.observe(identity,dict(prompt_tokens=4,completion_tokens=2))
            ledger.reserve('1'*64,10)
            self.assertTrue(Path(value['ledger_path']+'-wal').exists())
            before=ledger.summary()
            with patch.dict(os.environ,{},clear=True):report=snapshot(value,Path(directory)/'export')
            self.assertEqual(report['summary'],before)
            self.assertEqual(report['summary']['unresolved_reserved_nanousd'],1020000)
            self.assertEqual(report['summary']['known_token_price_nanousd'],8000)
            for name,digest in report['files'].items():self.assertEqual(hashlib.sha256((Path(directory)/'export'/name).read_bytes()).hexdigest(),digest)
            ledger.reserve('2'*64,10)
            exported=sqlite3.connect(Path(directory)/'export/ledger.sqlite')
            self.assertEqual(exported.execute('SELECT count(*) FROM calls').fetchone()[0],2);exported.close()
            self.assertEqual(ledger.summary()['requests'],3);ledger.close()
            with self.assertRaises(FileExistsError):snapshot(value,Path(directory)/'export')

    def test_missing_source_or_changed_policy_cannot_create_snapshot(self):
        from officegym.training.teacher_gateway import snapshot
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory);out=Path(directory)/'export'
            with self.assertRaises(sqlite3.OperationalError):snapshot(value,out)
            self.assertFalse(out.exists());self.assertFalse(Path(value['ledger_path']).exists())
            ledger=CostLedger(value);ledger.close();value['max_requests']+=1
            with self.assertRaisesRegex(ValueError,'policy differs'):snapshot(value,out)
            self.assertFalse(out.exists())

    @patch.dict(os.environ,{'TEST_TEACHER_LOCAL_KEY':'local-fixture-secret'})
    def test_deepseek_fixed_mode_and_tool_codec_roundtrip_preserves_provider_response(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory,endpoint='https://api.deepseek.com/chat/completions',max_input_tokens=4096,provider_options={'thinking':{'type':'disabled'}},tool_name_encoding='dots_to_double_underscore');observed=[]
            def transport(body,seconds):
                observed.append(body)
                result=response();result['choices'][0]['message']['tool_calls']=[dict(id='new',type='function',function=dict(name='office__get_context',arguments='{}'))]
                return result
            gateway=Gateway(value,transport);request=payload();request.update(parallel_tool_calls=False,tools=[dict(type='function',function=dict(name='office.get_context',description='read',parameters={'type':'object'}))],messages=[dict(role='assistant',content=None,tool_calls=[dict(id='old',type='function',function=dict(name='office.get_context',arguments='{}'))]),dict(role='tool',tool_call_id='old',content='{}')])
            result=gateway.request(request)
            self.assertEqual(observed[0]['thinking'],{'type':'disabled'});self.assertIs(observed[0]['parallel_tool_calls'],False);self.assertEqual(observed[0]['tools'][0]['function']['name'],'office__get_context');self.assertEqual(observed[0]['messages'][0]['tool_calls'][0]['function']['name'],'office__get_context')
            self.assertEqual(request['tools'][0]['function']['name'],'office.get_context')
            self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'],'office.get_context')
            self.assertEqual(result['_officegym_gateway']['provider_response']['choices'][0]['message']['tool_calls'][0]['function']['name'],'office__get_context')
            with self.assertRaisesRegex(ValueError,'unsupported billable'):gateway.request({**request,'thinking':{'type':'enabled'}})

    @patch.dict(os.environ,{'TEST_TEACHER_LOCAL_KEY':'local-fixture-secret'})
    def test_tool_codec_collision_fails_before_billable_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            value=config(directory,tool_name_encoding='dots_to_double_underscore');gateway=Gateway(value,lambda *args:self.fail('collision dispatched'))
            request=payload();request['tools']=[dict(type='function',function={'name':name}) for name in ('a.b','a__b')]
            with self.assertRaisesRegex(ValueError,'collision'):gateway.request(request)
            ledger=CostLedger(value)
            try:self.assertEqual(ledger.summary()['requests'],0)
            finally:ledger.close()

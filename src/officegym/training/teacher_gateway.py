"""Authenticated loopback gateway with conservative declared token-price budgets.

Operator-supplied prices and provider token limits are assumptions, not an invoice
or a universal tokenizer guarantee. Overruns are retained and stop future calls.
"""
import argparse
from decimal import Decimal,InvalidOperation,ROUND_CEILING,ROUND_FLOOR
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlparse
from urllib.request import Request,HTTPRedirectHandler,build_opener
from uuid import uuid4

NANO=10**9


def amount(value):
    if not isinstance(value,str):raise ValueError('USD values must be decimal strings')
    try:number=Decimal(value)
    except InvalidOperation as exc:raise ValueError('invalid USD amount') from exc
    if not number.is_finite() or number<0:raise ValueError('finite nonnegative USD amount required')
    return number


def price(policy,prompt,completion):
    value=(Decimal(prompt)*amount(policy['input_usd_per_million'])+Decimal(completion)*amount(policy['output_usd_per_million']))/10**6
    return int((value*NANO).to_integral_value(rounding=ROUND_CEILING))


def validate(config):
    fields=('endpoint','model','model_version','api_key_env','gateway_key_env','ledger_path','pricing','max_requests','max_input_tokens','max_output_tokens','max_wall_seconds','request_timeout_seconds')
    if any(key not in config for key in fields):raise ValueError('complete teacher gateway configuration required')
    endpoint=urlparse(config['endpoint'])
    if endpoint.scheme!='https' or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:raise ValueError('fixed HTTPS upstream without URL credentials/query required')
    for key in ('model','model_version'):
        if not isinstance(config[key],str) or not config[key] or config[key].startswith('SET_'):raise ValueError('operator-selected teacher identity required')
    for key in ('api_key_env','gateway_key_env'):
        if not isinstance(config[key],str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,127}',config[key]):raise ValueError('credential environment variable name required')
    if config['api_key_env']==config['gateway_key_env']:raise ValueError('local and upstream credentials must be separate')
    if not Path(config['ledger_path']).is_absolute():raise ValueError('absolute persistent API ledger required')
    for key in ('max_requests','max_input_tokens','max_output_tokens'):
        if type(config[key]) is not int or config[key]<1:raise ValueError('positive finite gateway request/token caps required')
    for key in ('max_wall_seconds','request_timeout_seconds'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key]<=0:raise ValueError('positive finite gateway wall/timeout caps required')
    if config['request_timeout_seconds']>config['max_wall_seconds']:raise ValueError('request timeout exceeds gateway lifetime')
    for key in ('max_cost_usd','input_usd_per_million','output_usd_per_million'):amount(config['pricing'][key])
    if amount(config['pricing']['max_cost_usd'])*NANO>2**63-1:raise ValueError('API budget exceeds ledger integer range')
    pricing=config['pricing']
    if pricing.get('usage_contract')!='total_billable_prompt_and_completion_tokens':raise ValueError('explicit total-billable-token usage contract required')
    if not isinstance(pricing.get('source'),str) or not pricing['source'] or pricing['source'].startswith('SET_'):raise ValueError('declared pricing source required')
    from datetime import datetime
    try:
        stamp=datetime.fromisoformat(pricing['as_of_utc'].replace('Z','+00:00'))
        if stamp.utcoffset() is None:raise ValueError('timezone required')
    except (KeyError,TypeError,AttributeError,ValueError) as exc:raise ValueError('dated timezone-aware pricing provenance required') from exc
    # One-nanodollar rounding is conservative for each observed/reserved call.
    return config


class CostLedger:
    def __init__(self,config):
        validate(config);self.config=config
        path=Path(config['ledger_path']);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY,config_sha TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY,body_sha TEXT,reserved INTEGER,output_cap INTEGER,observed INTEGER,usage TEXT,error TEXT,breach INTEGER DEFAULT 0)')
        fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.db:self.db.execute('INSERT OR IGNORE INTO policy VALUES (1,?)',(fingerprint,))
        self.policy_sha256=fingerprint
        if self.db.execute('SELECT config_sha FROM policy WHERE id=1').fetchone()[0]!=fingerprint:self.db.close();raise ValueError('cannot change existing teacher ledger policy')
        self.limit=int((amount(config['pricing']['max_cost_usd'])*NANO).to_integral_value(rounding=ROUND_FLOOR))

    def reserve(self,body_sha,output_limit):
        if not isinstance(body_sha,str) or not re.fullmatch('[0-9a-f]{64}',body_sha) or type(output_limit) is not int or not 1<=output_limit<=self.config['max_output_tokens']:raise ValueError('valid request hash and bounded output required')
        worst=price(self.config['pricing'],self.config['max_input_tokens'],output_limit)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count,charged,overrun=self.db.execute('SELECT count(*),coalesce(sum(coalesce(observed,reserved)),0),coalesce(max(breach),0) FROM calls').fetchone()
            if overrun or count>=self.config['max_requests'] or charged+worst>self.limit:raise ValueError('teacher API reservation budget exhausted or overrun')
            identity=uuid4().hex;self.db.execute('INSERT INTO calls (id,body_sha,reserved,output_cap) VALUES (?,?,?,?)',(identity,body_sha,worst,output_limit));self.db.commit();return identity
        except BaseException:self.db.rollback();raise

    def observe(self,identity,usage):
        if not isinstance(usage,dict) or any(type(usage.get(key)) is not int or usage[key]<0 for key in ('prompt_tokens','completion_tokens')):raise ValueError('missing measured provider token usage')
        cost=price(self.config['pricing'],usage['prompt_tokens'],usage['completion_tokens'])
        with self.db:
            row=self.db.execute('SELECT reserved,observed,output_cap FROM calls WHERE id=?',(identity,)).fetchone()
            if row is None or row[1] is not None:raise ValueError('unknown or already completed API reservation')
            breach=cost>row[0] or usage['prompt_tokens']>self.config['max_input_tokens'] or usage['completion_tokens']>row[2]
            self.db.execute('UPDATE calls SET observed=?,usage=?,breach=? WHERE id=?',(cost,json.dumps(usage,sort_keys=True),int(breach),identity))
        if breach:raise ValueError('provider exceeded declared billable token/price bound')

    def uncertain(self,identity,error,block=False):
        with self.db:self.db.execute('UPDATE calls SET error=?,breach=max(breach,?) WHERE id=?',(error,int(block),identity))

    def summary(self):
        count,known,unknown,reserved,overrun=self.db.execute('SELECT count(*),coalesce(sum(observed),0),coalesce(sum(CASE WHEN observed IS NULL THEN reserved ELSE 0 END),0),coalesce(sum(reserved),0),coalesce(max(breach),0) FROM calls').fetchone()
        missing=self.db.execute('SELECT count(*) FROM calls WHERE observed IS NULL').fetchone()[0]
        return dict(policy_sha256=self.policy_sha256,kind='declared-token-price-upper-bounds-not-invoice',requests=count,known_token_price_nanousd=known,unresolved_reserved_nanousd=unknown,charged_upper_nanousd=known+unknown,all_reserved_nanousd=reserved,max_cost_nanousd=self.limit,unresolved_requests=missing,provider_contract_breach=bool(overrun))

    def close(self):self.db.close()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,newurl):
        raise ValueError('teacher upstream redirects are not permitted')


class Gateway:
    def __init__(self,config,transport=None):
        self.config=validate(config);self.deadline=time.monotonic()+config['max_wall_seconds'];self.transport=transport or self.post;self.fixture=transport is not None
        if not os.environ.get(config['gateway_key_env']):raise ValueError('local gateway credential is unset')
        if transport is None and not os.environ.get(config['api_key_env']):raise ValueError('upstream teacher credential is unset')
        ledger=CostLedger(config);ledger.close()

    def post(self,payload,timeout):
        headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ[self.config['api_key_env']]}
        request=Request(self.config['endpoint'],data=json.dumps(payload,ensure_ascii=False).encode(),headers=headers,method='POST')
        with build_opener(NoRedirect()).open(request,timeout=timeout) as response:
            data=response.read(8*1024**2+1)
            if len(data)>8*1024**2:raise ValueError('upstream response byte cap')
            return json.loads(data)

    def request(self,payload):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise ValueError('teacher gateway lifetime exhausted')
        if not isinstance(payload,dict) or payload.get('model')!=self.config['model'] or payload.get('stream') is not False:raise ValueError('fixed model and nonstreaming request required')
        allowed={'model','messages','temperature','max_tokens','stream','tools','tool_choice','parallel_tool_calls'}
        if set(payload)-allowed:raise ValueError('unsupported billable API request fields')
        if payload.get('parallel_tool_calls',False) is not False:raise ValueError('serial function calls required')
        if payload.get('tool_choice','auto') not in ('auto','required','none'):raise ValueError('unsupported tool selection')
        tools=payload.get('tools',[])
        if not isinstance(tools,list) or any(not isinstance(tool,dict) or tool.get('type')!='function' for tool in tools):raise ValueError('only client-side function tools supported')
        output=payload.get('max_tokens')
        if type(output) is not int or not 1<=output<=self.config['max_output_tokens']:raise ValueError('teacher output cap violated')
        messages=payload.get('messages')
        if not isinstance(messages,list) or not messages:raise ValueError('teacher messages required')
        if any(not isinstance(message,dict) or set(message)-{'role','content','tool_calls','tool_call_id'} or message.get('role') not in ('system','user','assistant','tool') or (message.get('content') is not None and not isinstance(message['content'],str)) for message in messages):raise ValueError('text-only chat messages required')
        temperature=payload.get('temperature',0)
        if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=2:raise ValueError('finite sampling temperature required')
        # Conservative byte/framing guard. Provider token bounds remain a declared contract.
        encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
        framing=128*(len(messages)+len(payload.get('tools',[]))+1)
        if len(encoded)+framing>self.config['max_input_tokens']:raise ValueError('teacher serialized input/framing bound exceeded')
        ledger=CostLedger(self.config);identity=None
        try:
            identity=ledger.reserve(hashlib.sha256(encoded).hexdigest(),output)
            remaining=self.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('gateway lifetime exhausted before dispatch')
            response=self.transport(payload,min(remaining,self.config['request_timeout_seconds']))
            ledger.observe(identity,response.get('usage'))
            if response.get('model')!=self.config['model_version']:
                ledger.uncertain(identity,'ModelIdentityMismatch',block=True)
                raise ValueError('provider model identity differs from pinned teacher version')
            response=dict(response)
            response['_officegym_gateway']=dict(source_kind='software_fixture' if self.fixture else 'configured_upstream',request_id=identity,model_version=self.config['model_version'],pricing_policy_sha256=ledger.policy_sha256)
            return response
        except BaseException as error:
            if identity is not None:ledger.uncertain(identity,type(error).__name__)
            raise
        finally:ledger.close()

    def server(self,port=0):
        gateway=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                expected='Bearer '+os.environ[gateway.config['gateway_key_env']]
                if not hmac.compare_digest(self.headers.get('Authorization','').encode(),expected.encode()):self.send_error(401);return
                if self.path!='/budget':self.send_error(404);return
                ledger=CostLedger(gateway.config)
                try:data=json.dumps(ledger.summary()).encode()
                finally:ledger.close()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                self.connection.settimeout(min(10.,gateway.config['request_timeout_seconds']))
                expected='Bearer '+os.environ[gateway.config['gateway_key_env']]
                if not hmac.compare_digest(self.headers.get('Authorization','').encode(),expected.encode()):self.send_error(401);return
                if self.path!='/v1/chat/completions':self.send_error(404);return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=gateway.config['max_input_tokens']:raise ValueError('request byte cap')
                    response=gateway.request(json.loads(self.rfile.read(length)));data=json.dumps(response).encode()
                except Exception as error:
                    data=json.dumps(dict(error=dict(type=type(error).__name__,message='gateway rejected request; inspect budget ledger'))).encode();self.send_response(502)
                else:self.send_response(200)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--port',type=int,default=0);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());gateway=Gateway(config);server=gateway.server(args.port);server.timeout=.5
    print(json.dumps(dict(endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',model=config['model'],model_version=config['model_version'],cost_scope='declared token-price upper bounds')),flush=True)
    try:
        while time.monotonic()<gateway.deadline:server.handle_request()
    finally:server.server_close()


if __name__=='__main__':main()

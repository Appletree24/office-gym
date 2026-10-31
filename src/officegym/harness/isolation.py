"""Fail-closed Linux policy worker; trusted parent retains HTTP credentials/worlds."""
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from .loop import AssistantTurn
from .policy import ChatPolicy

FRAME_LIMIT = 16 * 1024 * 1024
ISOLATION_VERSION = 'policy-chroot-v1'


class IsolatedPolicy:
    policy_kind = 'inference_service'

    def __init__(self,policy,*,protected_paths=(),startup_seconds=10):
        if type(policy) is not ChatPolicy:
            raise ValueError('isolated entry accepts the canonical ChatPolicy only')
        if sys.platform!='linux' or os.geteuid()!=0:
            raise RuntimeError('Linux root with chroot/setuid capabilities required; no unsafe fallback')
        self.policy = policy
        self.policy_kind = policy.policy_kind
        for key in ('model_version','model','endpoint','temperature','max_output_tokens'):
            setattr(self,key,getattr(policy,key))
        self.process = None
        self.buffer = b''
        self.root = Path(tempfile.mkdtemp(prefix='officegym-policy-'))
        os.chmod(self.root,0o755)
        try:
            # No credential or caller environment is inherited. Imports occur before chroot.
            env = {'PATH':os.defpath,'LANG':'C.UTF-8','PYTHONPATH':str(Path(__file__).resolve().parents[2]),'PYTHONDONTWRITEBYTECODE':'1'}
            self.process = subprocess.Popen([sys.executable,'-m','officegym.harness.policy_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,bufsize=0,close_fds=True,start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(),False)
            os.set_blocking(self.process.stdout.fileno(),False)
            deadline = time.monotonic()+startup_seconds
            self._send(dict(method='initialize',root=str(self.root),protected_paths=[str(Path(p).absolute()) for p in protected_paths],model=self.model,model_version=self.model_version,max_output_tokens=self.max_output_tokens,temperature=self.temperature),deadline)
            response = self._receive(deadline)
            if response.get('kind')!='ready':
                raise RuntimeError('policy isolation setup failed: '+response.get('error_type','invalid handshake'))
            self.evidence = response['evidence']
            if not self.evidence.get('verified') or self.evidence.get('version')!=ISOLATION_VERSION:
                raise RuntimeError('policy isolation attestation failed')
        except BaseException:
            self.close()
            raise

    def _send(self,value,deadline):
        encoded = json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(encoded)>FRAME_LIMIT:
            raise ValueError('policy IPC request exceeds frame limit')
        offset = 0
        while offset<len(encoded):
            remaining = deadline-time.monotonic()
            if remaining<=0 or not select.select([], [self.process.stdin], [], max(0,remaining))[1]:
                raise TimeoutError('policy IPC write deadline')
            offset += os.write(self.process.stdin.fileno(),encoded[offset:])

    def _receive(self,deadline):
        while b'\n' not in self.buffer:
            remaining = deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],max(0,remaining))[0]:
                raise TimeoutError('policy IPC read deadline')
            chunk = os.read(self.process.stdout.fileno(),65536)
            if not chunk:
                raise RuntimeError('policy worker exited before response')
            self.buffer += chunk
            if len(self.buffer)>FRAME_LIMIT:
                raise ValueError('policy IPC response exceeds frame limit')
        line,self.buffer = self.buffer.split(b'\n',1)
        return json.loads(line)

    def _request(self,method,messages,remaining,schemas=None):
        deadline = time.monotonic()+remaining['wall_seconds']
        self._send(dict(method=method,messages=messages,remaining=remaining,schemas=schemas),deadline)
        transported = False
        while True:
            response = self._receive(deadline)
            if response.get('kind')=='transport' and not transported:
                transported = True
                try:
                    raw = self.policy.transport(response['payload'],max(.001,deadline-time.monotonic()))
                    reply = dict(kind='transport_result',raw=raw)
                except Exception as exc:
                    reply = dict(kind='transport_error',error_type=type(exc).__name__)
                self._send(reply,deadline)
            elif response.get('kind')=='result':
                return response['value']
            else:
                raise RuntimeError('policy worker failed: '+response.get('error_type','invalid response'))

    def generate(self,messages,schemas,remaining):
        return AssistantTurn(**self._request('generate',messages,remaining,schemas))

    def summarize(self,messages,remaining):
        return self._request('summarize',messages,remaining)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
        if getattr(self,'root',None):
            shutil.rmtree(self.root,ignore_errors=True)


def require_isolation(run_dir,result,*,allow_startup_failure=False):
    """Validate retained trusted-runner evidence before exporting/teaching."""
    evidence_path=Path(run_dir)/'policy_isolation.json'
    if not evidence_path.is_file():
        raise ValueError('missing policy isolation evidence')
    evidence=json.loads(evidence_path.read_text())
    if evidence!=result.get('policy_isolation') or evidence.get('version')!=ISOLATION_VERSION:
        raise ValueError('policy isolation evidence differs or is obsolete')
    if not evidence.get('verified'):
        startup_failure=result.get('infrastructure_error') and result.get('termination',{}).get('state')=='INFRA_ERROR' and evidence.get('status') in ('failed','not_started')
        startup_budget_stop=not result.get('infrastructure_error') and result.get('termination',{}).get('state')=='BUDGET_STOP' and evidence.get('status')=='budget_exhausted'
        if allow_startup_failure and (startup_failure or startup_budget_stop):
            trace=Path(run_dir)/'trajectory.jsonl'
            rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            if any(row.get('kind') in ('model_response','interaction') for row in rows):
                raise ValueError('unisolated infrastructure failure contains executed policy')
            return False
        raise ValueError('policy isolation was not verified')
    expected=dict(uid=65534,gid=65534,groups=[],no_new_privs=True,dumpable=0,cannot_regain_root=True,grader_unavailable=True,root_entries=[],inherited_descriptors=[0,1,2],parent_death_signal='SIGKILL',environment_empty=True)
    if any(evidence.get(key)!=value for key,value in expected.items()):
        raise ValueError('invalid policy isolation constraints')
    if not isinstance(evidence.get('protected_paths_denied'),list) or len(evidence['protected_paths_denied'])<5:
        raise ValueError('missing episode private path probes')
    return True

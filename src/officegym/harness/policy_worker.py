"""Exec-started policy worker. Receives public messages only, never worlds/grades."""
import ctypes
import fcntl
import signal
import json
import os
import sys
from dataclasses import asdict
from .isolation import FRAME_LIMIT,ISOLATION_VERSION
from .policy import ChatPolicy
from . import context  # Load summary code before the filesystem boundary.


def read():
    line = sys.stdin.buffer.readline(FRAME_LIMIT+1)
    if not line or len(line)>FRAME_LIMIT or not line.endswith(b'\n'):
        raise ValueError('invalid policy IPC frame')
    return json.loads(line)


def write(value):
    data = json.dumps(value,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(data)>FRAME_LIMIT:
        raise ValueError('policy IPC response too large')
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def restrict(root,paths):
    parent_pid=os.getppid()
    descriptors=[]
    for entry in os.listdir('/proc/self/fd'):
        try:
            fcntl.fcntl(int(entry),fcntl.F_GETFD)
        except OSError:
            continue
        descriptors.append(int(entry))
    if set(descriptors)!={0,1,2}:
        raise RuntimeError('unexpected inherited descriptor')
    libc = ctypes.CDLL(None,use_errno=True)
    # no_new_privs and nondumpable are reapplied after credentials change.
    if libc.prctl(38,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'no_new_privs')
    os.chdir(root)
    os.chroot('.')
    os.chdir('/')
    os.setgroups([])
    os.setresgid(65534,65534,65534)
    os.setresuid(65534,65534,65534)
    if libc.prctl(4,0,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'nondumpable')
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'parent death signal')
    if os.getppid()!=parent_pid:
        raise RuntimeError('parent exited during isolation setup')
    death_signal=ctypes.c_int()
    if libc.prctl(2,ctypes.byref(death_signal),0,0,0)!=0 or death_signal.value!=signal.SIGKILL:
        raise RuntimeError('parent death signal verification failed')
    os.environ.clear()
    denied=[]
    for path in paths:
        try:
            with open(path,'rb'):
                pass
        except OSError:
            denied.append(path)
        else:
            raise RuntimeError('protected file readable')
    try:
        os.setuid(0)
    except PermissionError:
        cannot_regain_root=True
    else:
        raise RuntimeError('root identity recoverable')
    try:
        import officegym.graders.result
    except ModuleNotFoundError:
        grader_unavailable=True
    else:
        raise RuntimeError('grader import available')
    evidence=dict(inherited_descriptors=sorted(descriptors),parent_death_signal='SIGKILL',environment_empty=not os.environ,version=ISOLATION_VERSION,uid=os.getuid(),gid=os.getgid(),groups=os.getgroups(),no_new_privs=libc.prctl(39,0,0,0,0)==1,dumpable=libc.prctl(3,0,0,0,0),protected_paths_denied=denied,cannot_regain_root=cannot_regain_root,grader_unavailable=grader_unavailable,root_entries=os.listdir('/'))
    evidence['verified']=evidence['uid']==65534 and evidence['gid']==65534 and not evidence['groups'] and evidence['no_new_privs'] and evidence['dumpable']==0 and not evidence['root_entries']
    return evidence


def bridge(payload,timeout):
    write(dict(kind='transport',payload=payload))
    response=read()
    if response.get('kind')!='transport_result':
        raise RuntimeError('trusted transport failed: '+response.get('error_type','invalid response'))
    return response['raw']


def main():
    try:
        init=read()
        if init['method']!='initialize':
            raise ValueError('initialization required')
        policy=ChatPolicy('broker://trusted-parent',init['model'],init['model_version'],max_output_tokens=init['max_output_tokens'],temperature=init['temperature'],transport=bridge)
        evidence=restrict(init['root'],init['protected_paths'])
        write(dict(kind='ready',evidence=evidence))
        while True:
            request=read()
            if request.get('method')=='generate':
                result=asdict(policy.generate(request['messages'],request['schemas'],request['remaining']))
            elif request.get('method')=='summarize':
                result=policy.summarize(request['messages'],request['remaining'])
            else:
                raise ValueError('unsupported policy method')
            write(dict(kind='result',value=result))
    except Exception as exc:
        write(dict(kind='error',error_type=type(exc).__name__))


if __name__=='__main__':
    main()

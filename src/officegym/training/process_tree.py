"""Linux subreaper supervision for jobs whose workers may call setsid().

Signals target only this dedicated supervisor's direct children. Their PIDs
cannot be reused before this process reaps them; no global Ray stop is used.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from officegym.eval.artifacts import write_json

CLEANUP_SECONDS=10


def children():
    result=[];owner=os.getpid()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
            if int(fields[1])==owner:
                result.append(dict(pid=int(path.name),state=fields[0],pgid=int(fields[2]),start_ticks=int(fields[19])))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result


def supervise(plan_path,result_path):
    plan=json.loads(Path(plan_path).read_text());budget=plan['max_wall_seconds']
    if type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0:raise ValueError('finite process-tree wall budget required')
    command=plan['command']
    if not isinstance(command,list) or not command or any(not isinstance(a,str) or not a for a in command):raise ValueError('explicit command argv required')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
    # Root owns no other child: every later direct child belongs to this job.
    if children():raise RuntimeError('dedicated supervisor already has children')
    signal.signal(signal.SIGCHLD,signal.SIG_DFL)
    cancelled=[]
    def cancel(signum,frame):cancelled.append(signum)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    parent=plan['parent_pid']
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'parent-death signal setup failed')
    if os.getppid()!=parent:cancelled.append(signal.SIGTERM)
    if cancelled:raise RuntimeError('launch parent exited before supervisor ownership setup')
    started=time.monotonic();process=subprocess.Popen(command,stdin=subprocess.DEVNULL,close_fds=True)
    primary_exit=None;exits=[];signalled=[]
    def reap():
        nonlocal primary_exit
        while True:
            try:pid,status=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:return
            if not pid:return
            code=os.waitstatus_to_exitcode(status);exits.append(dict(pid=pid,exit_code=code))
            if pid==process.pid:primary_exit=code;process.returncode=code
    while primary_exit is None and not cancelled and time.monotonic()-started<budget:
        reap()
        if primary_exit is None:time.sleep(.02)
    exhausted=primary_exit is None and not cancelled
    cleanup_started=time.monotonic()
    while True:
        reap();owned=children()
        if not owned:break
        # Do not reap between identifying and signalling direct children. Even
        # if they exit here, their zombie reserves the PID until our next reap.
        for row in owned:
            if row['state']=='Z':continue
            try:os.kill(row['pid'],signal.SIGSTOP)
            except ProcessLookupError:continue
            signalled.append(dict(row,signal='SIGSTOP'))
        for row in owned:
            if row['state']=='Z':continue
            try:os.kill(row['pid'],signal.SIGKILL)
            except ProcessLookupError:continue
            signalled.append(dict(row,signal='SIGKILL'))
        if time.monotonic()-cleanup_started>CLEANUP_SECONDS:break
        time.sleep(.02)
    reap();remaining=children()
    result=dict(supervision_version='linux-subreaper-v1',supervisor_pid=os.getpid(),command_pid=process.pid,exit_code=primary_exit,wall_budget_exhausted=exhausted,cancelled_signals=cancelled,process_tree_cleanup=not remaining,cleanup_seconds=time.monotonic()-cleanup_started,wall_seconds=time.monotonic()-started,signalled=signalled,reaped=exits,remaining=remaining)
    write_json(result_path,result)
    if remaining:raise RuntimeError('owned processes remain after cleanup; see identities')
    return result


def execute_tree_bounded(command,stdout,stderr,max_wall_seconds,*,evidence_dir,monitor=None,env=None):
    if type(max_wall_seconds) not in (int,float) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:raise ValueError('finite process-tree wall budget required')
    root=Path(evidence_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    plan=root/'plan.json';write_json(plan,dict(command=command,max_wall_seconds=max_wall_seconds,parent_pid=os.getpid()))
    result_path=root/'result.json'
    supervisor=subprocess.Popen([sys.executable,'-m','officegym.training.process_tree','--plan',str(plan),'--result',str(result_path)],stdout=stdout,stderr=stderr,start_new_session=True,env=env)
    write_json(root/'supervisor.json',dict(pid=supervisor.pid))
    try:
        # An observation overrun requests cancellation; it does not kill the
        # trusted reaper and abandon descendants.
        try:
            deadline=time.monotonic()+max_wall_seconds+CLEANUP_SECONDS+15
            while supervisor.poll() is None:
                if monitor is not None:monitor(supervisor.pid)
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(supervisor.args,max_wall_seconds)
                try:supervisor.wait(timeout=min(.25,remaining))
                except subprocess.TimeoutExpired:pass
        except subprocess.TimeoutExpired:
            supervisor.terminate();supervisor.wait(timeout=CLEANUP_SECONDS+15)
    finally:
        if supervisor.poll() is None:
            supervisor.terminate();supervisor.wait(timeout=CLEANUP_SECONDS+15)
    if not result_path.exists():raise RuntimeError('subreaper exited without ownership evidence')
    result=json.loads(result_path.read_text())
    if supervisor.returncode or not result.get('process_tree_cleanup'):raise RuntimeError('process-tree cleanup failed; inspect retained evidence')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--result',required=True);args=parser.parse_args()
    supervise(args.plan,args.result)

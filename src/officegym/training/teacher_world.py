"""Recompute teacher success from retained trusted inputs and actual world state."""
import hashlib
import json
import sqlite3
from pathlib import Path
from officegym.eval.grading import require_recorded_grade
from .data import RejectedTrajectory

INPUTS=('task.json','private_spec.json','event_rules.json','initial_world.sqlite')


def verify_teacher_world(root,result):
    root=Path(root)
    try:
        if not (root/'world.sqlite').is_file():raise ValueError('missing retained final world')
        hashes=result['versions']['instance_artifact_sha256']
        if any(hashes.get(name)!=hashlib.sha256((root/name).read_bytes()).hexdigest() for name in INPUTS):
            raise ValueError('retained input hash mismatch')
        task=json.loads((root/'task.json').read_text());private=json.loads((root/'private_spec.json').read_text())
        if task.get('split')!='train' or any(task.get(key)!=result.get(key) for key in ('instance_id','split','template_id')):
            raise ValueError('retained training task identity mismatch')
        grade,audit=require_recorded_grade(root/'world.sqlite',{**task['public_task'],**private},root/'initial_world.sqlite',result)
        if not grade['full_success'] or not audit['consistent']:raise ValueError('retained world is not a verified success')
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as exc:
        raise RejectedTrajectory('teacher_world_evidence_invalid') from exc
    return grade


def verify_teacher_trace(root,records):
    """Deterministically replay all recorded observations on a disposable world."""
    import tempfile
    from officegym.env.rollout import clone_world
    from officegym.env.world import canonical
    from officegym.harness.loop import initial_messages
    from officegym.tools.dispatcher import Dispatcher
    from officegym.tools.schema import tool_schemas,WRITES
    root=Path(root)
    try:
        task=json.loads((root/'task.json').read_text())['public_task'];header=records[0]
        if header['tool_schemas']!=tool_schemas() or header['initial_messages']!=initial_messages(task,full=True) or header['task_sha256']!=hashlib.sha256(canonical(task).encode()).hexdigest():
            raise ValueError('teacher prompt/schema differs from retained public task')
        cap=header['budget']['tool_calls']
        if type(cap) is not int or cap<1:raise ValueError('finite original tool cap required')
        with tempfile.TemporaryDirectory(prefix='officegym-teacher-replay-') as temporary:
            world=clone_world(root/'initial_world.sqlite',Path(temporary)/'world.sqlite')
            try:
                # Match trusted run_episode startup, which overrides the initial
                # task's default tool cap before starting the world service.
                with world.db:world.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('max_tool_calls',canonical(cap)))
                client=Dispatcher(world,max_calls=cap);pending=None;count=0;operations=set()
                for row in records[1:]:
                    if row['kind']=='action_start':
                        if pending is not None:raise ValueError('overlapping teacher actions')
                        pending=row
                    elif row['kind']=='interaction':
                        if pending is None or row['action_id']!=pending['action_id'] or row.get('operation_id')!=pending.get('operation_id'):
                            raise ValueError('unmatched teacher replay action')
                        operation=pending.get('operation_id');tool=pending['tool']
                        if tool in WRITES:
                            if not isinstance(operation,str) or not operation or operation in operations:raise ValueError('unique original write operation required')
                            operations.add(operation)
                        elif operation is not None:raise ValueError('read/control action cannot have a write operation')
                        expected=client.call(tool,pending['arguments'],operation)
                        messages=row['messages']
                        if messages[0]['role']!='assistant' or messages[1]['role']!='tool' or json.loads(messages[1]['content'])!=expected:
                            raise ValueError('teacher observation differs from actual tool replay')
                        pending=None;count+=1
                if pending is not None or count<1:raise ValueError('incomplete teacher replay')
                final=sqlite3.connect((root/'world.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
                try:
                    final.row_factory=sqlite3.Row
                    tables=[row[0] for row in final.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
                    snapshot={table:sorted((dict(row) for row in final.execute('SELECT * FROM "'+table+'"')),key=canonical) for table in tables}
                    if world.snapshot()!=snapshot:raise ValueError('teacher replay differs from retained final world')
                finally:final.close()
                return dict(replayed_tool_calls=count,final_state_sha256=world.state_hash())
            finally:world.close()
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error,IndexError) as exc:
        raise RejectedTrajectory('teacher_trace_replay_invalid') from exc

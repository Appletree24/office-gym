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

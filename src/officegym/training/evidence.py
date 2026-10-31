"""Append-only training curves and content-addressed saved-checkpoint inventory."""
import hashlib
import math
import os
import time
from pathlib import Path
from officegym.env.world import canonical
from officegym.eval.artifacts import write_json


def finite_logs(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {'non_finite':str(value)}
    if isinstance(value,dict):
        return {key:finite_logs(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_logs(item) for item in value]
    return value


class TrainingEvidence:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.started=time.monotonic()

    def event(self,kind,step,payload):
        if type(step) is not int or step<0:
            raise ValueError('nonnegative optimizer step required')
        record=dict(kind=kind,optimizer_step=step,elapsed_seconds=time.monotonic()-self.started,payload=finite_logs(payload))
        with (self.root/'training_events.jsonl').open('a') as stream:
            stream.write(canonical(record)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def checkpoint(self,path,step,kind='training_checkpoint'):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError('saved checkpoint must be inside training artifacts')
        hashes={}
        for file in sorted(path.rglob('*')):
            if file.is_file():
                if not file.resolve().is_relative_to(path):
                    raise ValueError('saved checkpoint contains external symlink')
                digest=hashlib.sha256()
                with file.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                        digest.update(chunk)
                hashes[str(file.relative_to(path))]=digest.hexdigest()
        if not hashes or not any(name.endswith(('.safetensors','.bin')) for name in hashes):
            raise ValueError('saved checkpoint has no model weights')
        record=dict(path=str(path),files_sha256=hashes,checkpoint_files_sha256=hashlib.sha256(canonical(hashes).encode()).hexdigest())
        self.event(kind,step,record)
        return record

    def history(self,records):
        write_json(self.root/'trainer_log_history.json',finite_logs(records))

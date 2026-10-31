"""Durable atomic JSON records for episode setup and terminal results."""
import os
from pathlib import Path
from uuid import uuid4
from officegym.env.world import canonical


def write_json(path,value):
    path = Path(path)
    payload = (canonical(value)+'\n').encode()
    temporary = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

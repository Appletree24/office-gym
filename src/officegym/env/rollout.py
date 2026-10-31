"""Trusted per-rollout database lifecycle, preserving live WAL snapshot contents."""
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from .world import World


def clone_world(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive reservation prevents accidental overwrite of another rollout.
    with destination.open('xb'):
        pass
    reader = writer = None
    try:
        reader = sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
        writer = sqlite3.connect(destination)
        reader.backup(writer)
        if writer.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('invalid snapshot clone')
    except BaseException:
        if writer:
            writer.close()
            writer = None
        destination.unlink(missing_ok=True)
        raise
    finally:
        if reader:
            reader.close()
        if writer:
            writer.close()
    return World(destination)


@contextmanager
def rollout_world(initial_path, *, work_dir=None):
    """Grade inside this scope, before automatic cleanup. One directory per rollout."""
    with tempfile.TemporaryDirectory(prefix='officegym_rollout_',dir=work_dir) as directory:
        world = clone_world(initial_path,Path(directory)/'world.sqlite')
        try:
            yield world
        finally:
            world.close()

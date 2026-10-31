"""Versioned runtime contracts shipped with the Python distribution."""
from importlib.resources import files


def read_text(name):
    if name not in ('world.sql','runtime.sql','tools.json'):
        raise ValueError('unknown OfficeGym runtime resource')
    return files(__package__).joinpath(name).read_text(encoding='utf-8')

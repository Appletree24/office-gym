#!/usr/bin/env python3
"""Compatibility entrypoint for the packaged OfficeGym statistics implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from officegym.eval.analysis import *

if __name__=='__main__':
    main()

"""Resolve external compute Python separately from a frozen controller."""
import os
from pathlib import Path
import shutil
import sys

def suggested_python():
    configured=os.environ.get('PI_TRAINER_TRAINING_PYTHON')
    if configured:return configured
    suffix='Scripts/python.exe' if os.name=='nt' else 'bin/python'
    candidates=[Path(__file__).resolve().parent.parent/'.venv-training'/suffix,Path.cwd()/'.venv-training'/suffix]
    for candidate in candidates:
        if candidate.is_file():return str(candidate)
    if not getattr(sys,'frozen',False):return sys.executable
    return shutil.which('python3.11') or shutil.which('python3') or shutil.which('python') or 'python3.11'

def compute_python(spec):
    value=str(spec.get('python') or suggested_python())
    if getattr(sys,'frozen',False) and Path(value).resolve()==Path(sys.executable).resolve():
        raise ValueError('Select an external Python environment with the required training or Hailo SDK packages')
    return str(Path(value).absolute()) if Path(value).is_file() else value

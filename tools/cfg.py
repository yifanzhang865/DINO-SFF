"""Load a Python config with the original py2cfg interface."""
import runpy
from types import SimpleNamespace

def py2cfg(path):
    return SimpleNamespace(**{k:v for k,v in runpy.run_path(str(path)).items() if not k.startswith('__')})

"""Shared provenance and storage checks for separately declared release fits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_lab.storage import GIB, require_space


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_path(name):
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Declared paths must stay inside the repository.')
    return path


def space(protocol,destination,expected):
    roots = [repository_path(protocol[name]) for name in ('raw_root','training_package','model_package','evidence_package')]
    used = sum(path.stat().st_size for root in roots if root.exists() for path in root.rglob('*') if path.is_file())
    used += sum(path.stat().st_size for path in (ROOT/'.run/final-goal').glob(roots[0].name+'*') if path.is_file())
    if used+expected > protocol['budget']['maximum_data_gib_including_packages']*GIB:
        raise ValueError('The declared raw-plus-packaged data budget would be exceeded.')
    return require_space(destination,expected,protocol['budget']['reserve_gib']*GIB)


def write_json(path,value,*,replace=False):
    path = Path(path)
    require_space(path,4*1024**2)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not replace:
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(value,indent=2)+'\n')
    else:
        pending = path.with_suffix(path.suffix+'.pending')
        pending.write_bytes((json.dumps(value,indent=2)+'\n').encode())
        pending.replace(path)

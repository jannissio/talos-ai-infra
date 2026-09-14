"""Build a small, credential-free Windows verification kit from tracked files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def sha(value):
    return hashlib.sha256(value).hexdigest()


def build(args):
    output, archive = args.output.resolve(), args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve previous bundles and archives.')
    names = subprocess.check_output(['git', 'ls-files', '--', 'simulation_lab/*.py', 'scripts/*.py', 'simulation_lab/assets/so101'], cwd=ROOT, text=True).splitlines()
    names += ['LICENSE', 'simulation_lab/NOTICE.md', 'verify-final-on-intel.ps1', 'requirements.txt',
              'requirements-training.txt', 'requirements-openvino.txt']
    model_files = {'primitive.json', 'primitive.safetensors', 'primitive.xml', 'primitive.bin', 'visual.npz',
                   'talos_normalization.json', 'openvino.json', 'benchmark.json', 'README.md', 'suite.json'}
    for folder in ('bottle_visual', 'bottle_relays', 'dinner_suite'):
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'models'/folder).rglob('*') if p.is_file() and p.name in model_files]
    names = sorted(set(names))
    if any(Path(name).is_absolute() or '..' in Path(name).parts or any(part in ('.git', '.run', '__pycache__') or part.startswith('.env') for part in Path(name).parts) for name in names):
        raise ValueError('Private/generated paths cannot enter the allowlisted bundle.')
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    if any(name not in tracked for name in names):
        raise ValueError('Only tracked, reviewed inputs belong in the kit.')
    total = sum((ROOT/name).stat().st_size for name in names)
    if total > 64*1024**2:
        raise ValueError('The verification kit exceeded its 64 MiB uncompressed payload bound.')
    preflight = require_space(output, total+4*1024**2)
    require_space(archive, total+4*1024**2)
    output.mkdir(parents=True)
    mapping = {}
    for name in names:
        data = (ROOT/name).read_bytes()
        require_space(output/name, len(data)+1024)
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        mapping[name] = sha(data)
    readme = '''# Talos Intel verification kit

This kit contains the exact selected dinner/relay models, tracked source, robot
assets and MIT/Apache notices. It requires the existing clean Python 3.12 training
environment with MuJoCo, PyTorch and OpenVINO. It contains no environment, API key,
microphone recording or training dataset. No installation or upload occurs.

Extract into a NEW folder. From that folder in PowerShell, use your laptop's
existing Python executable:

```powershell
$talosPython = Read-Host "Full path to the existing .venv-training\\Scripts\\python.exe"
& .\\verify-final-on-intel.ps1 -Python $talosPython -Output ".run/intel-final-laptop-cpu"
```

The wrapper checks Python/dependencies and free space, then benchmarks actual
OpenVINO devices and runs all six learned skills on exposed seed 42. Keep the
output directory unused. It retains a 10 GiB reserve plus estimated writes and
does not overwrite models or previous evidence.

On the historical i7-10850H laptop the strict Core Ultra check remains false even
if physics succeeds. Read both PhysicalSequencePassed and
StrictHardwareAndPhysicsPassed. No switch can change the measured CPU/graphics
identity or establish organizer eligibility.

Retain the entire .run/intel-final-laptop-cpu folder, especially verification.json,
hardware.json, physical.json, all benchmark.json files and physical-recording/.
Share the verification results with the Talos task for review. No Speechmatics key
or microphone is needed. Do not put credentials in this kit or send them in chat.

kit-manifest.json records every original source/model/asset hash and the source
Git revision. The separate kit audit records a PC portability check; AMD/NVIDIA
execution does not substitute for actual Intel execution.

The original project and this archive remain private until the final release.
Only the user presses the hackathon's final Submit button.
'''
    for name, data in [('README-VERIFICATION.md', readme.encode()), ('.gitignore', b'.run/\n.venv*/\n.env\n.env.*\n__pycache__/\n*.py[cod]\n')]:
        require_space(output/name, len(data)+1024)
        with (output/name).open('xb') as stream:
            stream.write(data)
    files = {path.relative_to(output).as_posix(): sha(path.read_bytes()) for path in output.rglob('*') if path.is_file()}
    report = {'schema': 'talos.intel-verification-kit.v1', 'source_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'builder_sha256': sha(Path(__file__).read_bytes()), 'original_file_sha256': mapping, 'files': files,
        'uncompressed_bytes': sum((output/name).stat().st_size for name in files), 'storage_preflight': preflight,
        'scope': 'Portable source/model preparation. Actual target-hardware execution and eligibility remain separate.'}
    require_space(output/'kit-manifest.json', 1024**2)
    with (output/'kit-manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    require_space(archive, sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+1024**2)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                require_space(archive, path.stat().st_size+1024)
                info = zipfile.ZipInfo(path.relative_to(output).as_posix(), date_time=(2026, 9, 14, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(json.dumps({'files': len(files)+1, 'uncompressed_bytes': report['uncompressed_bytes'],
                      'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive.read_bytes()),
                      'source_git_revision': report['source_git_revision']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    build(parser.parse_args())

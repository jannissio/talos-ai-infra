"""Fetch the published MIT CLIP RN50 backbone explicitly; never overwrite files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space

SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
URL = f'https://openaipublic.azureedge.net/clip/models/{SHA256}/RN50.pt'


def fetch(folder):
    if folder.exists():raise FileExistsError('Preserve the existing download or failed attempt.')
    preflight = require_space(folder, 512*1024**2); folder.mkdir(parents=True)
    report = {'url': URL, 'expected_sha256': SHA256, 'preflight': preflight,
              'role': 'Frozen vision/language representation, not a pretrained robotics controller.'}
    digest = hashlib.sha256(); size = 0
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, (folder/'RN50.pt').open('xb') as stream:
            while chunk := response.read(4*1024**2):
                require_space(folder, len(chunk)+8*1024**2)
                stream.write(chunk); digest.update(chunk); size += len(chunk)
        report.update(bytes=size, sha256=digest.hexdigest(), passed=digest.hexdigest() == SHA256)
        if not report['passed']:raise ValueError('CLIP weight digest does not match the published hash.')
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc), bytes=size)
        raise
    finally:
        with (folder/'download.json').open('x', encoding='utf-8') as stream:json.dump(report, stream, indent=2)
        print({k: v for k, v in report.items() if k != 'url'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    fetch(parser.parse_args().output)

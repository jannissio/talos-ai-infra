"""Freeze the passing bottle observer before the single reserved perception split."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, space, write
from scripts.export_bottle_refinement import score_runtime
from simulation_lab.bottle_refinement_runtime import RefinedBottleObserver


def run(args):
    protocol = args.protocol.resolve()
    p = checked_protocol(protocol)
    raw = ROOT / p['raw_root']
    freeze_path = raw / 'fresh-perception-freeze.json'
    result_path = raw / 'fresh-perception.json'
    failure_path = raw / 'fresh-perception-failure.json'
    logs = [raw / 'evaluation-collect.log', raw / 'evaluation-audit.log']
    if any(path.exists() for path in [freeze_path, result_path, failure_path, raw / 'evaluation', *logs]):
        raise FileExistsError('Preserve every reserved evaluation attempt and its console logs.')
    selection = read(raw / 'fit/selection.json')
    export = read(raw / 'openvino/parity-development.json')
    if not selection['development_gate_passed'] or not export['passed']:
        raise ValueError('The declared development and CPU export gates must both pass.')
    if selection['protocol_sha256'] != sha(protocol) or export['protocol_sha256'] != sha(protocol):
        raise ValueError('Development/export results belong to a different protocol.')
    checkpoint = ROOT / selection['selected']['checkpoint']
    if sha(checkpoint) != selection['selected']['checkpoint_sha256']:
        raise ValueError('The selected checkpoint changed.')
    if sha(raw / 'openvino/artifacts.json') != export['artifacts_sha256']:
        raise ValueError('The exported runtime artifacts changed.')
    inputs = [raw / 'fit/selection.json', checkpoint, raw / 'openvino/parity-development.json',
              raw / 'openvino/artifacts.json']
    for split in ('training', 'development'):
        folder = raw / split
        audit = read(folder / 'audit.json')
        if not audit['passed'] or audit['protocol_sha256'] != sha(protocol):
            raise ValueError('All preceding data audits must pass before fresh exposure.')
        if audit['data_manifest_sha256'] != sha(folder / 'manifest.json'):
            raise ValueError('An audited data manifest changed.')
        inputs += [folder / 'audit.json', folder / 'manifest.json']
    sources = ['scripts/evaluate_bottle_refinement.py', 'scripts/export_bottle_refinement.py',
               'scripts/audit_bottle_refinement_data.py', 'scripts/bottle_refinement_experiment.py',
               'simulation_lab/bottle_refinement_runtime.py', 'simulation_lab/bottle_refinement.py']
    frozen = {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}
    frozen.update({name: sha(ROOT / name) for name in sources})
    preflight = space(p, raw / 'evaluation', 256 * 1024**2)
    # Construction also checks every runtime source and both OpenVINO networks.
    torch.set_num_threads(2)
    runtime = RefinedBottleObserver(raw / 'openvino')
    write(freeze_path, {'schema': p['schema'], 'protocol_sha256': sha(protocol),
        'frozen_sha256': frozen, 'preflight': preflight,
        'evaluation_seed': p['data']['evaluation_rng_seed'],
        'selection_rule': 'Previously selected checkpoint and unchanged declared confidence/error limits.',
        'scope': 'One reserved perception evaluation; no physical success or deployment claim.'})
    stage = 'collection'
    began = time.perf_counter()
    try:
        commands = [
            [sys.executable, '-I', str(ROOT / 'scripts/bottle_refinement_experiment.py'),
             '--protocol', str(protocol), '--action', 'collect', '--split', 'evaluation'],
            [sys.executable, '-I', str(ROOT / 'scripts/audit_bottle_refinement_data.py'),
             '--protocol', str(protocol), '--split', 'evaluation'],
        ]
        for index, (command, log) in enumerate(zip(commands, logs)):
            stage = 'collection' if index == 0 else 'data_audit'
            if log.exists() or (index == 0 and (raw / 'evaluation').exists()):
                raise FileExistsError('A reserved output or console log already exists.')
            space(p, log, 256 * 1024**2 if index == 0 else 8 * 1024**2)
            with log.open('xb') as stream:
                outcome = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=p['budget']['maximum_generation_minutes_per_split']*60 + 60 if index == 0 else 1200)
            print({'stage': stage, 'exit_code': outcome.returncode}, flush=True)
            if outcome.returncode:
                raise RuntimeError('Reserved perception '+stage+' failed; preserve its original log.')
        for name, digest in frozen.items():
            if sha(ROOT / name) != digest:
                raise ValueError('A frozen input changed: '+name)
        stage = 'cpu_evaluation'
        result = score_runtime(raw / 'evaluation', p, runtime, sha(protocol))
        result.update(protocol_sha256=sha(protocol), freeze_sha256=sha(freeze_path),
            evaluation_manifest_sha256=sha(raw / 'evaluation/manifest.json'),
            evaluation_audit_sha256=sha(raw / 'evaluation/audit.json'),
            evaluator_sha256=sha(Path(__file__)), passed=result['summary']['gate_passed'],
            wall_seconds=time.perf_counter()-began,
            scope='All reserved synthetic RGB states; not a physical manipulation result.')
        write(result_path, result)
        print(result['summary'], flush=True)
        return int(not result['passed'])
    except Exception as exc:
        write(failure_path, {'stage': stage, 'type': type(exc).__name__, 'message': str(exc),
            'freeze_sha256': sha(freeze_path), 'wall_seconds': time.perf_counter()-began,
            'action': 'Stop and preserve this attempt; do not relaunch on the reserved split.'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))

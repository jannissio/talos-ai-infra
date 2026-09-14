"""Export the selected refiner and score every development RGB observation on CPU."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import openvino as ov
import torch
from safetensors.torch import load_file
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, write, space
from simulation_lab.bottle_refinement import KeypointCropRefiner, decode
from simulation_lab.bottle_refinement_runtime import RefinedBottleObserver
from simulation_lab.rgb_servo_cameras import VIEWS
from simulation_lab.rgb_servo_network import BottleKeypointNet, decode_heatmaps


def score_runtime(folder, p, runtime, protocol_sha):
    manifest = read(folder/'manifest.json')
    if manifest['protocol_sha256'] != protocol_sha or manifest['completed_states'] != manifest['states']:
        raise ValueError('Dataset does not match the declared complete split.')
    calibrations = dict(zip(VIEWS, manifest['calibrations']))
    rows = []
    for shard in manifest['shards']:
        path = folder/shard['file']; assert sha(path) == shard['sha256']
        with np.load(path, allow_pickle=False) as z:
            for index in range(len(z['rgb'])):
                observed = runtime.observe(dict(zip(VIEWS, z['rgb'][index])), calibrations)
                present = bool(z['present'][index])
                truth = z['world_points'][index, 1]
                error = float(np.linalg.norm(np.asarray(observed['grasp_point_m'])-truth)*1000) if present and observed['status'] == 'observed' else None
                rows.append({'index': len(rows), 'scoring_only_present': present, 'scoring_only_grasp_m': truth.tolist() if present else None,
                    'observation': observed, 'error_mm': error})
        print({'scored_rgb_states': len(rows), 'total': manifest['states']}, flush=True)
    errors = [r['error_mm'] for r in rows if r['error_mm'] is not None]
    present = sum(r['scoring_only_present'] for r in rows)
    false = sum(not r['scoring_only_present'] and r['observation']['status'] == 'observed' for r in rows)
    summary = {'states': len(rows), 'present': present, 'accepted_present': len(errors), 'absent': len(rows)-present,
        'false_accepted_absent': false, 'accepted_fraction': len(errors)/max(1, present),
        'p95_error_mm': float(np.quantile(errors, .95)) if errors else None, 'maximum_error_mm': max(errors) if errors else None}
    gate = p['perception_gate']
    summary['gate_passed'] = bool(errors and summary['accepted_fraction'] >= gate['minimum_present_acceptance_fraction']
        and false <= gate['maximum_false_absent_accepts'] and summary['p95_error_mm'] <= gate['maximum_p95_error_mm']
        and summary['maximum_error_mm'] <= gate['maximum_error_mm'])
    return {'summary': summary, 'rows': rows}


def run(args):
    p = checked_protocol(args.protocol); raw = ROOT/p['raw_root']; output = raw/'openvino'
    if output.exists():
        raise FileExistsError('Preserve prior exports.')
    selection = read(raw/'fit/selection.json'); chosen = selection['selected']
    if not selection['development_gate_passed'] or selection['protocol_sha256'] != sha(args.protocol):
        raise ValueError('No frozen passing development selection.')
    checkpoint = ROOT/chosen['checkpoint']
    assert sha(checkpoint) == chosen['checkpoint_sha256']
    torch.set_num_threads(2)
    model = KeypointCropRefiner().eval(); model.load_state_dict(load_file(str(checkpoint)))
    sample = torch.zeros(6, 4, 64, 64)
    traced = torch.jit.trace(model, sample)
    converted = ov.convert_model(traced, example_input=(sample,))
    space(p, output, 32*1024**2); output.mkdir()
    ov.save_model(converted, str(output/'refiner.xml'), compress_to_fp16=False)
    coarse_root = ROOT/'docs/robotics/evidence/rgb-servo-observer-camera-v1/validation/openvino'
    for ext in ('xml', 'bin'):
        write(output/('coarse.'+ext), (coarse_root/('observer.'+ext)).read_bytes())
    write(output/'protocol.json', args.protocol.read_bytes())
    source_names = ['simulation_lab/bottle_refinement.py', 'simulation_lab/bottle_refinement_runtime.py',
        'simulation_lab/rgb_servo_network.py', 'simulation_lab/rgb_servo_cameras.py', 'simulation_lab/rgb_servo_geometry.py']
    artifact = {'schema': p['schema'], 'selected_checkpoint_sha256': sha(checkpoint),
        'coarse_checkpoint_sha256': sha(ROOT/p['coarse_checkpoint']),
        'files': {name: sha(output/name) for name in ('coarse.xml', 'coarse.bin', 'refiner.xml', 'refiner.bin', 'protocol.json')},
        'runtime_sources': {name: sha(ROOT/name) for name in source_names},
        'exporter_sha256': sha(Path(__file__)), 'scope': 'Experimental CPU exports. Physical promotion is a separate gate.'}
    write(output/'artifacts.json', artifact)
    runtime = RefinedBottleObserver(output)
    coarse_model = BottleKeypointNet().eval()
    coarse_model.load_state_dict(load_file(str(ROOT/p['coarse_checkpoint'])))
    coarse_deltas = []
    first_shard = read(raw/'development/manifest.json')['shards'][0]['file']
    with np.load(raw/'development'/first_shard, allow_pickle=False) as z:
        for index in np.linspace(0, len(z['rgb'])-1, 12).astype(int):
            rgb = z['rgb'][index].transpose(0, 3, 1, 2).astype(np.float32)/255
            with torch.inference_mode():
                reference = decode_heatmaps(coarse_model(torch.from_numpy(rgb))[0])[0].numpy()
                actual = decode_heatmaps(torch.from_numpy(runtime.coarse([rgb])[0].copy()))[0].numpy()
            coarse_deltas.append(float(np.max(np.abs(reference-actual))))
    deltas = []
    with np.load(raw/'prepared-development/crops.npz', allow_pickle=False) as z:
        indices = np.linspace(0, len(z['crops'])-6, 12).astype(int)
        for begin in indices:
            inputs = z['crops'][begin:begin+6].astype(np.float32)
            with torch.inference_mode():
                reference = decode(model(torch.from_numpy(inputs)))[0].numpy()
                actual = runtime.refiner([inputs])
                observed = decode(tuple(torch.from_numpy(actual[i].copy()) for i in range(2)))[0].numpy()
            deltas.append(float(np.max(np.abs(reference-observed))))
    result = score_runtime(raw/'development', p, runtime, sha(args.protocol))
    original = read((ROOT/chosen['checkpoint']).parent/'development.json')
    decision_changes = sum(a['observation']['status'] != b['observation']['status'] for a, b in zip(original['rows'], result['rows']))
    result.update(protocol_sha256=sha(args.protocol), artifacts_sha256=sha(output/'artifacts.json'),
        maximum_decoded_pixel_parity=max(deltas), decoded_pixel_parity_probes=deltas,
        coarse_decoded_pixel_parity_probes=coarse_deltas,
        selection_decision_changes=decision_changes, exporter_sha256=sha(Path(__file__)),
        passed=max(deltas) <= .01 and max(coarse_deltas) <= .01 and result['summary']['gate_passed'])
    write(output/'parity-development.json', result)
    print({'passed': result['passed'], 'pixel_parity': max(deltas), 'development': result['summary'], 'decision_changes': decision_changes}, flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))

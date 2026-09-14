"""Export a development-qualified mug observer and check all development inputs."""
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
from simulation_lab.mug_visual_observer import MugShapePoseNet
from scripts.mug_observer_experiment import load_protocol, read, score, sha, space, write_json
from scripts.train_mug_visual_observer import dataset


def run(args):
    p = load_protocol(args.protocol)
    raw = ROOT/p['raw_root']
    selection = read(raw/'fit/selection.json')
    if not selection['development_gate_passed'] or selection['protocol_sha256'] != sha(args.protocol):
        raise ValueError('Export requires frozen successful development.')
    checkpoint = ROOT/selection['checkpoint']
    if sha(checkpoint) != selection['checkpoint_sha256'] or sha(raw/'fit/training.json') != selection['training_report_sha256']:
        raise ValueError('The selected fit changed.')
    output = raw/'openvino'
    if output.exists():
        raise FileExistsError('Preserve all previous exports.')
    preflight = space(p, output, 16*1024**2)
    source, manifest = dataset(raw/'development', p, args.protocol)
    torch.set_num_threads(2)
    model = MugShapePoseNet().eval()
    model.load_state_dict(load_file(str(checkpoint)))
    traced = torch.jit.trace(model, torch.zeros(1, p['observer']['feature_dimensions']))
    converted = ov.convert_model(traced, example_input=(torch.zeros(1, p['observer']['feature_dimensions']),))
    output.mkdir(parents=True)
    space(p, output, 8*1024**2)
    ov.save_model(converted, output/'observer.xml', compress_to_fp16=False)
    core = ov.Core()
    compiled = core.compile_model(str(output/'observer.xml'), 'CPU',
        {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    with torch.inference_mode():
        expected = model(torch.from_numpy(source['features'])).numpy().astype(float)
    actual, latencies = [], []
    for feature in source['features']:
        started = time.perf_counter()
        actual.append(compiled([feature[None, :]])[0][0].copy())
        latencies.append((time.perf_counter()-started)*1000)
    actual = np.asarray(actual, dtype=float)
    scale, origin = p['observer']['target_scale_m'], np.asarray(p['observer']['target_origin_m'])
    error = float(np.max(abs(actual-expected))*scale)
    development = score(p, actual*scale+origin, source['labels_m'], source['present'], source['usable_views'])
    parity_passed = error <= p['export']['maximum_absolute_parity_error_m']
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'source_sha256': sha(Path(__file__)),
        'checkpoint_sha256': sha(checkpoint), 'selection_sha256': sha(raw/'fit/selection.json'),
        'ir_sha256': {name: sha(output/name) for name in ('observer.xml', 'observer.bin')},
        'development_data_sha256': manifest['data_sha256'], 'inputs': len(expected),
        'maximum_absolute_parity_error_m': error, 'parity_passed': parity_passed,
        'development': development, 'passed': parity_passed and development['summary']['gate_passed'],
        'inference_median_ms': float(np.median(latencies)), 'all_inference_ms': latencies,
        'device': core.get_property('CPU', 'FULL_DEVICE_NAME'), 'execution_devices': list(compiled.get_property('EXECUTION_DEVICES')),
        'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT')), 'openvino_version': ov.__version__,
        'preflight': preflight, 'scope': 'FP32 numerical and exposed development checks on this PC. No Intel or physical control claim.'}
    space(p, output, 2*1024**2)
    write_json(output/'parity.json', result)
    print({key: result[key] for key in ('maximum_absolute_parity_error_m', 'parity_passed', 'passed', 'inference_median_ms', 'device')})
    if not result['passed']:
        raise ValueError('Export failed its declared parity/development gate; preserve it.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-observer-v1.json')
    run(parser.parse_args())

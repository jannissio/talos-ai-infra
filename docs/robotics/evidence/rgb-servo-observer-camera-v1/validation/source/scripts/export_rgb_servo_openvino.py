"""Export selected RGB and motor networks; verify numerical FP32 parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import openvino as ov
import torch
from safetensors.torch import load_file
from simulation_lab.rgb_servo_network import BottleKeypointNet, decode_heatmaps
from simulation_lab.rgb_servo_motor import CartesianMotorNet
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, 64*1024**2); args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    observer = BottleKeypointNet().eval(); observer.load_state_dict(load_file(str(args.observer)))
    motor = CartesianMotorNet().eval(); motor.load_state_dict(load_file(str(args.motor)))
    core = ov.Core(); results = {}; rng = np.random.default_rng(2026095601)
    for name, model, sample in [('observer', observer, (torch.zeros(3, 3, 240, 320),)),
                                ('motor', motor, (torch.zeros(1, 3), torch.zeros(1)))]:
        require_space(args.output, 32*1024**2)
        traced = torch.jit.trace(model, sample)
        converted = ov.convert_model(traced, example_input=sample)
        path = args.output/(name+'.xml'); ov.save_model(converted, path, compress_to_fp16=False)
        compiled = core.compile_model(str(path), 'CPU', {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
        errors = []; decoded = []
        for index in range(12):
            if name == 'observer':
                inputs = [torch.from_numpy(rng.random((3, 3, 240, 320), dtype=np.float32))]
            else:
                inputs = [torch.tensor(rng.uniform([-.14, -.18, .888], [.16, -.06, .970], (1, 3)), dtype=torch.float32),
                          torch.tensor([-1. if index % 2 else 1.])]
            with torch.inference_mode():
                expected = model(*inputs)
            actual = compiled([tensor.numpy() for tensor in inputs])
            expected = (expected,) if isinstance(expected, torch.Tensor) else expected
            errors.append(max(float(np.max(np.abs(tensor.numpy()-actual[output]))) for tensor, output in zip(expected, compiled.outputs)))
            if name == 'observer':
                a = decode_heatmaps(expected[0])[0].numpy()
                b = decode_heatmaps(torch.from_numpy(actual[compiled.outputs[0]].copy()))[0].numpy()
                decoded.append(float(np.max(np.abs(a-b))))
        passed = max(errors) < (.001 if name == 'observer' else .00001)
        results[name] = {'maximum_absolute_output_error': max(errors), 'maximum_decoded_pixel_error': max(decoded) if decoded else None,
                         'parity_passed': passed, 'samples': len(errors), 'input_scope': 'Deterministic numerical probes, not physical evaluation.',
                         'device': core.get_property('CPU', 'FULL_DEVICE_NAME'), 'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT'))}
        if not passed:
            raise ValueError(name+' export failed numerical parity.')
    result = {'observer_sha256': hashlib.sha256(args.observer.read_bytes()).hexdigest(),
              'motor_sha256': hashlib.sha256(args.motor.read_bytes()).hexdigest(), 'openvino_version': ov.__version__,
              'networks': results, 'scope': 'AMD PC export/parity check, not Intel execution evidence or physical regression.'}
    require_space(args.output, 1024**2)
    (args.output/'parity.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observer', type=Path, required=True); p.add_argument('--motor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

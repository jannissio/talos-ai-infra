"""Export the actual LeRobot ACT inference boundary and check Intel CPU numerical parity."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import openvino as ov
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulation_lab.act_learning import IMAGE_KEYS, load_policy


class Inference(torch.nn.Module):
    def __init__(self, policy, stats):
        super().__init__()
        self.policy = policy
        self.relative = stats.get('action_representation') == 'relative'
        for name, key in [('state', 'observation.state'), ('action', 'action')]:
            for kind in ['mean', 'std']:
                self.register_buffer(name + '_' + kind, torch.tensor(stats[key][kind], dtype=torch.float32))
        for kind in ['mean', 'std']:
            self.register_buffer('image_' + kind, torch.tensor(stats['images'][kind], dtype=torch.float32)[None, :, None, None])

    def forward(self, state, overhead, left_wrist, right_wrist):
        batch = {'observation.state': (state - self.state_mean) / self.state_std}
        batch.update({key: (value - self.image_mean) / self.image_std
                      for key, value in zip(IMAGE_KEYS, (overhead, left_wrist, right_wrist))})
        result = self.policy.predict_action_chunk(batch) * self.action_std + self.action_mean
        return result + state[:, None, :12] if self.relative else result


def main(a):
    if (Path(a.checkpoint) / 'primitive.safetensors').exists():
        raise ValueError('Motion primitives require their own export and joint-feedback wrapper; this exporter supports ACT only.')
    if (Path(a.checkpoint) / 'sequence.safetensors').exists():
        raise ValueError('A recurrent model needs an explicit hidden-state export boundary; this exporter is ACT-only.')
    if (Path(a.checkpoint) / 'retrieval.npz').exists():
        raise ValueError('NumPy retrieval is not traceable ACT; export would freeze observation-dependent behavior.')
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Use a fresh export folder.')
    output.mkdir(parents=True)
    checkpoint = Path(a.checkpoint)
    policy, stats = load_policy(checkpoint, 'cpu')
    wrapper = Inference(policy, stats).eval()
    ds = LeRobotDataset('talos/local', root=Path(a.dataset), video_backend='pyav')
    def inputs(index):
        row = ds[index]
        return tuple(row[key].unsqueeze(0) for key in ['observation.state', *IMAGE_KEYS])
    example = inputs(0)
    print('Tracing frozen CPU inference boundary.', flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, check_trace=True)
    converted = ov.convert_model(traced, example_input=example)
    names = ['state', 'overhead', 'left_wrist', 'right_wrist']
    for port, name in zip(converted.inputs, names):
        port.get_tensor().set_names({name})
    converted.reshape({name: list(value.shape) for name, value in zip(names, example)})
    converted.output(0).get_tensor().set_names({'joint_targets'})
    ov.save_model(converted, output / 'act.xml', compress_to_fp16=False)
    core = ov.Core()
    compiled = core.compile_model(str(output / 'act.xml'), 'CPU',
        {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2, 'PERFORMANCE_HINT': 'LATENCY'})
    errors = []
    for index in np.unique(np.linspace(0, len(ds) - 1, a.samples, dtype=int)):
        sample = inputs(int(index))
        with torch.inference_mode():
            reference = wrapper(*sample).numpy()
        actual = compiled({name: value.numpy() for name, value in zip(names, sample)})[0]
        if actual.shape != (1, 20, 12) or not np.isfinite(actual).all():
            raise ValueError('Invalid exported policy output.')
        errors.append(float(np.max(np.abs(reference - actual))))
    report = {'checkpoint': str(checkpoint), 'dataset': a.dataset,
        'checkpoint_sha256': hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
        'openvino_version': ov.__version__, 'available_devices': core.available_devices,
        'execution_devices': compiled.get_property('EXECUTION_DEVICES'),
        'cpu_name': core.get_property('CPU', 'FULL_DEVICE_NAME'),
        'samples': len(errors), 'max_abs_joint_target_error_rad': max(errors),
        'input_shapes': {port.get_any_name(): str(port.get_partial_shape()) for port in compiled.inputs},
        'per_sample_max_error_rad': errors, 'tolerance_rad': 1e-4,
        'passed': max(errors) <= 1e-4,
        'input_contract': 'Float32 raw motor state [1,24], three RGB [1,3,240,320] in [0,1]. Normalization is embedded.',
        'output_contract': 'Float32 nominal joint targets [1,20,12] in radians. Servo limits remain external.',
        'precision': 'FP32 weights and CPU inference hint; no quantization',
        'scope': 'Numerical export/inference check only; no physical task success or timing benchmark claimed.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Export failed numerical parity tolerance.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', default='.run/act-matched-nvidia-fit5')
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=16)
    a = p.parse_args()
    if a.samples < 1:
        p.error('--samples must be positive')
    main(a)

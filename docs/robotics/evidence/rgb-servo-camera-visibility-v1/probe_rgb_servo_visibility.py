"""Compare preregistered camera placements on exposed saved approach states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration
from simulation_lab.rgb_servo_openvino import OpenVinoBottleObserver
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostics; choose a fresh output directory.')
    require_space(args.output, protocol['budget']['maximum_new_data_mib']*1024**2)
    model_folder = ROOT / protocol['observer_model']
    if sha(model_folder / 'observer.safetensors') != protocol['observer_sha256']:
        raise ValueError('Observer weights changed.')
    for name, digest in protocol['observer_ir_sha256'].items():
        if sha(model_folder / 'openvino' / name) != digest:
            raise ValueError('Observer IR changed: ' + name)
    torch.set_num_threads(2)
    observer = OpenVinoBottleObserver(model_folder / 'openvino/observer.xml', minimum_views=2, rigid_geometry=True)
    args.output.mkdir(parents=True)
    began, rows, inputs = time.perf_counter(), [], {}
    for trace in protocol['traces']:
        folder = ROOT / protocol['input_root'] / trace
        for name in ('scene.xml', 'states.npz', 'report.json'):
            inputs[(folder / name).relative_to(ROOT).as_posix()] = sha(folder / name)
        with np.load(folder / 'states.npz', allow_pickle=False) as source:
            states = {key: source[key] for key in source.files}
        end = min(protocol['maximum_simulation_time_s'], float(states['time'][-1]))
        indices = np.unique([int(np.argmin(np.abs(states['time']-t)))
                             for t in np.linspace(0, end, protocol['frames_per_trace'])])
        model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
        data = mujoco.MjData(model)
        model.vis.quality.offsamples = 0
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        renderer = mujoco.Renderer(model, width=320, height=240)
        try:
            for index in indices:
                if time.perf_counter()-began > protocol['budget']['maximum_compute_minutes']*60:
                    raise TimeoutError('Declared diagnostic compute budget reached; retain partial output.')
                require_space(args.output, 4*1024**2)
                data.qpos[:] = states['qpos'][index]
                data.qvel[:] = states['qvel'][index]
                data.time = float(states['time'][index])
                mujoco.mj_forward(model, data)
                for name, settings in protocol['camera_configurations'].items():
                    images, cameras = {}, {}
                    for slot in VIEWS:
                        camera = camera_argument(slot)
                        camera.azimuth, camera.elevation, camera.distance = settings[slot]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                        images[slot] = renderer.render().copy()
                        cameras[slot] = calibration(renderer)
                    result = observer.observe(images, cameras)
                    # Ground truth is used only after image-only inference, never as its input.
                    truth = KEYPOINTS @ data.body('bottle').xmat.reshape(3,3).T + data.body('bottle').xpos
                    error = float(np.linalg.norm(np.asarray(result['keypoints_m'])-truth, axis=1).max()*1000) if result['status']=='observed' else None
                    rows.append({'trace':trace, 'state_index':int(index), 'simulation_seconds':float(data.time),
                                 'stage':str(states['stage'][index]), 'configuration':name, 'observation':result,
                                 'calibrations':cameras, 'scoring_only_error_mm':error})
                    if index == indices[-1]:
                        image_path = args.output / (trace+'-'+name+'.png')
                        require_space(image_path, 1024**2)
                        Image.fromarray(np.concatenate([images[slot] for slot in VIEWS], axis=1)).save(image_path)
        finally:
            renderer.close()
        require_space(args.output / 'partial.json', 8*1024**2)
        (args.output / 'partial.json').write_text(json.dumps(rows)+'\n', encoding='utf-8')
        print(json.dumps({'completed_trace':trace, 'observations':len(rows)}), flush=True)
    baseline = {(r['trace'],r['state_index']):r['observation']['status']=='observed'
                for r in rows if r['configuration']=='original'}
    missing = {key for key, accepted in baseline.items() if not accepted}
    gate, summaries = protocol['diagnostic_gate'], {}
    for name in protocol['camera_configurations']:
        subset = [r for r in rows if r['configuration']==name]
        accepted = [r for r in subset if r['observation']['status']=='observed']
        errors = [r['scoring_only_error_mm'] for r in accepted]
        recovered = sum((r['trace'],r['state_index']) in missing for r in accepted)
        p95, maximum = (float(np.quantile(errors,.95)), max(errors)) if errors else (None,None)
        summaries[name] = {'frames':len(subset), 'accepted':len(accepted), 'original_refused_frames':len(missing),
                           'recovered_original_refusals':recovered, 'accepted_error_p95_mm':p95, 'accepted_error_max_mm':maximum,
                           'diagnostic_gate_passed': bool(missing and errors and len(accepted)/len(subset)>=gate['minimum_overall_acceptance_fraction']
                              and recovered/len(missing)>=gate['minimum_recovery_fraction_on_original_refused_frames']
                              and p95<=gate['maximum_accepted_p95_error_mm'] and maximum<=gate['maximum_accepted_error_mm'])}
    eligible = [name for name in summaries if name!='original' and summaries[name]['diagnostic_gate_passed']]
    selected = min(eligible,key=lambda name:(-summaries[name]['recovered_original_refusals'],summaries[name]['accepted_error_p95_mm'])) if eligible else None
    dependencies = json.loads((ROOT / 'docs/robotics/evidence/rgb-servo-v5/physical/development/frozen-inputs.json').read_text())['source_sha256']
    if any(sha(ROOT / name)!=digest for name,digest in dependencies.items()):
        raise ValueError('A frozen runtime dependency changed during diagnosis.')
    report = {'schema':protocol['schema'], 'protocol_sha256':sha(args.protocol), 'source_sha256':sha(Path(__file__)),
              'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'runtime_source_sha256':dependencies, 'input_sha256':inputs, 'summaries':summaries,
              'selected_configuration':selected, 'wall_seconds':time.perf_counter()-began, 'rows':rows,
              'scope':'Exposed saved-state image diagnostic only. No new physical trial, neural training, fresh perception test or promotion.'}
    require_space(args.output / 'report.json', 8*1024**2)
    (args.output / 'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    for original, name in [(args.protocol,'protocol.json'), (Path(__file__),'probe_rgb_servo_visibility.py')]:
        require_space(args.output / name, original.stat().st_size + 1024**2)
        (args.output / name).write_bytes(original.read_bytes())
    generated = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
    if generated > protocol['budget']['maximum_new_data_mib']*1024**2:
        raise ValueError('Diagnostic data budget exceeded; preserve output and do not select.')
    print(json.dumps({'summaries':summaries,'selected_configuration':selected,'wall_seconds':report['wall_seconds'],'generated_bytes':generated}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-visibility-v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

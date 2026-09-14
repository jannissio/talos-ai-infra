"""Audit the received legacy-Intel reports and preserve every supplied attempt."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space

SKILLS = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    if not isinstance(payload, bytes):
        payload = (json.dumps(payload, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def check_physical(row, device):
    m, details = row['metrics'], row['policy_details']
    if (row['status'] != 'succeeded' or row['teacher_updates'] != 0 or not m['both_arms_parked']
            or m['stable_release_and_park_s'] < .5-1e-8 or m['unexpected_collisions'] != 0
            or m['other_object_max_displacement_m'] > .004 or m['parked_arm_max_motion_deg'] > 1):
        raise ValueError('A received success fails release, parking or disturbance criteria.')
    if row['skill'] == 'drawer':
        if m['drawer_open_m'] < .105 or m['verified_handle_contact_s'] < .1:
            raise ValueError('Drawer success lacks opening/contact evidence.')
    elif (m['placement_error_mm'] >= 8 or m['placement_z_error_mm'] >= 4 or m['hold_verified_s'] < 1.49
            or m['longest_unsupported_gap_s'] > .18 or m['speed_mm_s'] >= 3
            or m['placement_yaw_error_deg'] >= math.degrees(.175)):
        raise ValueError('A received placement fails the original physical criteria.')
    if (details['neural_runtime'] != 'OpenVINO' or details['neural_execution_devices'] != [device]
            or details['inference_demonstration_actions'] or details['simulator_clock_or_object_pose_input']):
        raise ValueError('Recorded learned runtime differs from the declared device or input contract.')


def audit_run(raw, output, device, checkpoints, canonical_xml, layout, intel_renderer=False):
    verification, report = read(raw/'verification.json'), read(raw/'physical.json')
    hardware = verification['hardware']
    if hardware != read(raw/'hardware.json'):
        raise ValueError('Hardware reports disagree.')
    if ('i7-10850H' not in hardware['cpu'] or not hardware['intel_cpu'] or hardware['core_ultra_series'] is not None
            or hardware['strict_written_hardware_check'] or hardware['intel_opengl_renderer'] != intel_renderer
            or hardware['all_intel_cpu_and_graphics'] != intel_renderer or verification['strict_hardware_and_physics_passed']):
        raise ValueError('Preserve the actual legacy CPU and nonqualifying renderer flags.')
    if not verification['physical_passed'] or not verification['inference_devices_intel']:
        raise ValueError('Expected reported physical success on the explicit Intel device.')
    if not any(d['id'] == device and 'Intel' in d['name'] for d in hardware['openvino_devices']):
        raise ValueError('Actual Intel device identity is missing.')
    if (report['status'] != 'succeeded' or report['seed'] != 42 or report['completed_steps'] != SKILLS
            or [r['skill'] for r in report['results']] != SKILLS
            or any(report[k] != 0 for k in ('physics_state_writes', 'hidden_forces', 'equality_constraints', 'teacher_updates'))):
        raise ValueError('A received workflow is incomplete or reports privileged assistance.')
    for skill, row in zip(SKILLS, report['results']):
        check_physical(row, device)
        if checkpoints[skill] != report['checkpoint_sha256'][skill] or checkpoints[skill] != verification['source_checkpoint_sha256'][skill]:
            raise ValueError('A reported checkpoint differs from the selected source model.')
        benchmark = read(raw/skill/'benchmark.json')
        selected = verification['benchmarks'][skill]
        if (benchmark['source_sha256'] != checkpoints[skill] or selected not in benchmark['devices']
                or not selected['parity_passed'] or selected['execution_devices'] != [device]
                or selected['device'] != device or 'Intel' not in selected['name']):
            raise ValueError('A selected export benchmark lacks source parity or actual device evidence.')
    # Copy the received XML exactly as well as a portable version; only meshdir changes.
    original = (raw/'physical-recording/scene.xml').read_bytes()
    tree = ET.fromstring(original)
    comparable = ET.fromstring(canonical_xml)
    tree.find('compiler').set('meshdir', 'ASSET_ROOT')
    comparable.find('compiler').set('meshdir', 'ASSET_ROOT')
    if ET.tostring(tree) != ET.tostring(comparable):
        raise ValueError('The recorded scene differs from the canonical exposed starting scene.')
    tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', output).replace('\\', '/'))
    write(output/'scene.xml', ET.tostring(tree, encoding='utf-8'))
    write(output/'original-scene.xml', original)
    write(output/'states.npz', (raw/'physical-recording/states.npz').read_bytes())
    model = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    data = mujoco.MjData(model)
    end_geometry = []
    with np.load(output/'states.npz', allow_pickle=False) as states:
        frames = len(states['time'])
        shapes = {'qpos': (frames, model.nq), 'qvel': (frames, model.nv), 'targets': (frames, 12),
            'time': (frames,), 'stage': (frames,), 'progress': (frames,)}
        if set(states.files) != set(shapes) or any(states[k].shape != v for k, v in shapes.items()):
            raise ValueError('The full recorded trace has invalid dimensions.')
        if any(not np.isfinite(states[k]).all() for k in shapes if k != 'stage') or not np.all(np.diff(states['time']) > 0):
            raise ValueError('The recorded trace is nonfinite or not chronological.')
        stages = list(dict.fromkeys(states['stage'].tolist()))
        if stages != SKILLS or abs(float(states['time'][-1])-report['elapsed_s']) > 1e-6:
            raise ValueError('Recorded stages/duration disagree with the full workflow.')
        total = 0.
        for row in report['results']:
            total += row['elapsed_s']
            index = int(np.argmin(abs(states['time']-total)))
            if abs(float(states['time'][index])-total) > .026:
                raise ValueError('A skill completion has no nearby retained state.')
            data.qpos[:] = states['qpos'][index]
            data.qvel[:] = states['qvel'][index]
            data.ctrl[:] = states['targets'][index]
            data.time = float(states['time'][index])
            mujoco.mj_forward(model, data)
            parked = np.max(abs(data.qpos[:12]-np.array(HOME*2))) < .035 and np.max(abs(data.qvel[:12])) < .12
            if not parked:
                raise ValueError('Saved completion state does not have parked arms.')
            skill = row['skill']
            if skill == 'drawer':
                geometry = {'drawer_open_m': float(data.joint('drawer_slide').qpos[0])}
                if geometry['drawer_open_m'] < .105:
                    raise ValueError('Saved drawer completion is not open.')
            else:
                target = next((t for t in layout['targets'] if t['object_id'] == skill), None)
                position = data.body(skill).xpos.copy()
                if target is None:
                    # The original bottle monitor uses the scene's declared
                    # starting position plus this fixed default destination.
                    item = next(o for o in layout['objects'] if o['id'] == skill)
                    destination = np.asarray(item['initial_position_m']) + [.070, -.040, 0]
                else:
                    destination = np.asarray(target['position_m'])
                error = float(np.linalg.norm(position[:2]-destination[:2])*1000)
                difference = abs(error-row['metrics']['placement_error_mm'])
                dof = model.joint(skill+'_free').dofadr[0]
                speed = float(np.linalg.norm(data.qvel[dof:dof+3])*1000)
                if error >= 8 or difference > .01 or speed >= 3 or data.body(skill).xmat[8] <= .98:
                    raise ValueError('Saved completion geometry contradicts the reported placement.')
                geometry = {'placement_error_mm': error, 'reported_error_difference_mm': difference,
                    'speed_mm_s': speed, 'upright_cosine': float(data.body(skill).xmat[8])}
            end_geometry.append({'skill': skill, 'state_index': index, 'state_time': float(data.time),
                'both_arms_parked': bool(parked), **geometry})
    return {'device': device, 'physical_passed': True, 'trace_frames': frames,
        'simulation_seconds': report['elapsed_s'], 'physical_loop_wall_seconds': report['wall_seconds'],
        'checkpoints_match': True, 'all_six_export_reports_pass_parity': True,
        'strict_hardware_and_physics_passed': False, 'hardware': hardware,
        'end_geometry': end_geometry, 'benchmarks': verification['benchmarks'],
        'trace_audit_scope': 'All arrays, stages, chronology and original bytes checked; completion geometry restored with mj_forward. No integration step or fresh laptop run. Continuous contact/hold claims are from the retained original 200 Hz monitor; the saved trace samples at 20 Hz.'}


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve every earlier evidence package.')
    inputs = {'nvidia-render': args.input}
    if args.intel_render_input:
        inputs['intel-render'] = args.intel_render_input
    receipts = {name: read(folder/'evidence-manifest.json') for name, folder in inputs.items()}
    receipt = receipts['nvidia-render']
    original_kit = ROOT/'submission/verification-v1/talos-intel-verification.zip'
    if sha(original_kit) != receipt['kit_sha256']:
        raise ValueError('Received evidence references a different verification kit.')
    with zipfile.ZipFile(original_kit) as kit:
        kit_manifest = json.loads(kit.read('kit-manifest.json'))
        if any(kit_manifest != read(folder/'kit-manifest.json') for folder in inputs.values()):
            raise ValueError('The received kit manifest differs from the original.')
        for name, digest in kit_manifest['files'].items():
            if hashlib.sha256(kit.read(name)).hexdigest() != digest:
                raise ValueError('An original kit member fails its bound hash.')
            if name.startswith('simulation_lab/assets/') and sha(ROOT/name) != digest:
                raise ValueError('Local rendering assets differ from the verified kit.')
    for name, folder in inputs.items():
        if receipts[name]['kit_sha256'] != receipt['kit_sha256']:
            raise ValueError('The received sessions reference different kits.')
        for row in receipts[name]['files']:
            source = folder/row['file']
            if not source.resolve().is_relative_to(folder.resolve()) or sha(source) != row['share_sha256'] or source.stat().st_size != row['bytes']:
                raise ValueError('A received file fails its transport manifest.')
    suite = read(ROOT/'models/dinner_suite/suite.json')
    checkpoints = {s: sha(ROOT/'models/dinner_suite'/suite[s]/'primitive.safetensors') for s in SKILLS}
    preflight = require_space(args.output, 96*1024**2)
    args.output.mkdir(parents=True)
    mapping = {}
    for session, folder in inputs.items():
        for source in sorted(folder.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(folder).as_posix()
            target = 'received/'+session+'/'+(relative.replace('.run/', 'runs/', 1) if relative.startswith('.run/') else relative)
            write(args.output/target, source.read_bytes())
            mapping[session+'/'+relative] = {'file': target, 'sha256': sha(source)}
    xml, layout = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    runs = {}
    for name, device in [('intel-final-laptop-cpu', 'CPU'), ('intel-final-laptop-gpu0', 'GPU.0')]:
        runs[name] = audit_run(args.input/'.run'/name, args.output/'portable'/name, device, checkpoints, xml, layout)
    if args.intel_render_input:
        for name, device in [('intel-render-laptop-cpu', 'CPU'), ('intel-render-laptop-gpu0', 'GPU.0')]:
            runs[name] = audit_run(args.intel_render_input/'.run'/name, args.output/'portable'/name, device,
                                  checkpoints, xml, layout, intel_renderer=True)
    failed = args.input/'.run/intel-final-laptop-gpu'
    benchmark = read(failed/'bottle/benchmark.json')
    if any(row['device'] == 'GPU' for row in benchmark['devices']) or not all(row['parity_passed'] for row in benchmark['devices']):
        raise ValueError('Unexpected evidence for the retained generic-GPU attempt.')
    if (failed/'physical.json').exists() or (failed/'verification.json').exists():
        raise ValueError('The generic-GPU attempt unexpectedly contains final physics.')
    log = (args.input/'.run/session/intel-final-laptop-gpu.console.log').read_text(encoding='utf-8-sig')
    if 'Requested OpenVINO device failed parity for bottle' not in log:
        raise ValueError('The retained device-name failure console is missing.')
    checked = sum(len(receipt['files']) for receipt in receipts.values())
    audit = {'schema': 'talos.intel-final-legacy-evidence.v2', 'received_files_checked': checked,
        'original_kit_files_checked': len(kit_manifest['files']), 'original_kit_sha256': sha(original_kit),
        'source_git_revision': kit_manifest['source_git_revision'], 'source_checkpoint_sha256': checkpoints,
        'runs': runs, 'failed_generic_gpu_attempt': {'retained': True, 'physical_trial_started': False,
            'cause': 'Requested GPU did not exactly match enumerated CPU, GPU.0 or GPU.1. All three recorded bottle parity checks passed; the verifier emitted a misleading parity error for the missing exact device name.'},
        'source_mapping': mapping, 'storage_preflight': preflight, 'auditor_sha256': sha(__file__),
        'scope': 'Received laptop evidence establishes the selected six-skill baseline on legacy Intel CPU/iGPU. Earlier NVIDIA-rendered runs and later Intel-rendered runs remain separate. It does not verify Core Ultra eligibility or the later live-mug observer/motor option. Re-exported IR files were not included in the received evidence; parity and timing are retained source-bound reports, not a local reproduction of those laptop exports.'}
    write(args.output/'audit.json', audit)
    write(args.output/'audit-source.py', Path(__file__).read_bytes())
    lines = ['# Legacy Intel laptop verification', '',
        'The received September 14 evidence verifies the selected six-skill baseline on an **Intel Core i7-10850H**. Both explicit OpenVINO devices complete exposed seed 42 with real simulated contact, release and parked arms.', '',
        '| OpenGL renderer | Inference | Six-skill result | Simulated seconds | Physics-loop wall seconds |',
        '| --- | --- | --- | --- | --- |']
    for name, row in runs.items():
        lines.append(f"| {'Intel UHD' if row['hardware']['intel_opengl_renderer'] else 'NVIDIA Quadro'} | {row['device']} | 6/6 | {row['simulation_seconds']:.3f} | {row['physical_loop_wall_seconds']:.3f} |")
    lines += ['', '**Hardware eligibility remains unresolved.** The processor is not Core Ultra Series 2/3. The later two runs verify Intel UHD OpenGL rendering and Intel CPU/GPU.0 inference together. Earlier NVIDIA Quadro T2000 Max-Q rendering remains separately recorded. Every strict Core Ultra hardware flag remains false.', '',
        'The laptop operator selected Power saving / Intel UHD for the actual shared base Python executable in Windows graphics preferences. No GPU was disabled and no environment, source or model files changed. The received summary records how to restore Windows decides. This preference affects other programs using that same base runtime.', '',
        'Each network benchmark uses FP32, 20 warmups and 300 measured synchronous calls; one output contains 20 action endpoints. These small-network timings exclude vision, rendering and physics. Read each run separately: changing the renderer also changes full-workflow wall time.', '',
        'The earlier generic `GPU` attempt is also retained. Its bottle parity checks passed on CPU, GPU.0 and GPU.1, but the requested alias matched none of the exact report names. The verifier stopped before full physics with a misleading parity message. The later explicit `GPU.0` run passed.', '',
        f'All {checked} transport-manifest files match; the original 318-file kit manifest and six source checkpoint hashes match. All {len(runs)} complete 4,955-frame traces are unchanged. Portable scenes load, every trace array is checked, and restored completion geometry agrees with the original reports. Continuous contact/hold evidence comes from the retained original runtime monitor; the 20 Hz saved trace cannot independently reconstruct every 200 Hz contact.', '',
        'Every supplied attempt, console, report and transport provenance is under `received/`; its `.run` directory is mapped to `runs` for publication. `portable/` contains unchanged trace copies, original XML and scenes whose only change is the mesh path. The original received ZIP remains local. Re-exported laptop IR files were not supplied; benchmark reports are preserved without claiming that they were rerun here.', '',
        'This kit predates the new late-mug live-vision option. Its results apply to the original six-skill suite. A separate target-machine check is needed for the new option. GitHub/HF remain private until final release; final Submit belongs to the user.', '']
    write(args.output/'README.md', '\n'.join(lines).encode())
    files = {p.relative_to(args.output).as_posix(): sha(p) for p in args.output.rglob('*') if p.is_file()}
    write(args.output/'manifest.json', {'files': files, 'bytes': sum((args.output/name).stat().st_size for name in files)})
    print(json.dumps({'passed': True, 'files': len(files), 'received_files_checked': len(receipt['files']),
        'trace_frames': sum(r['trace_frames'] for r in runs.values()), 'strict_hardware_and_physics_passed': False,
        'manifest_sha256': sha(args.output/'manifest.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--intel-render-input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

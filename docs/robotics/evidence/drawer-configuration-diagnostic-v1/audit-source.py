"""Preserve all exposed drawer diagnoses and verify their physical records."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
from scripts.measure_manipulation_coverage import read, sha, put
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


def run():
    protocol = ROOT/'docs/robotics/experiments/drawer-configuration-diagnostic-v1.json'
    p = read(protocol); raw = ROOT/p['raw_root']; output = ROOT/p['evidence_package']
    summary = read(raw/'summary.json')
    assert summary['all_15_cases_retained'] and len(summary['rows']) == 15
    assert output.is_relative_to(ROOT) and not output.exists()
    preflight = require_space(output, sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())+32*1024**2)
    copied = {}
    def copy(source, target):
        require_space(target, source.stat().st_size+1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        copied[target.relative_to(output).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source in raw.rglob('*'):
        if source.is_file():
            relative = source.relative_to(raw)
            if source.name == 'scene.xml':
                relative = relative.parent/'source-scene.xml'
            copy(source, output/relative)
    copy(ROOT/'.run/final-goal/drawer-configuration-diagnostic-v1.log', output/'console.log')
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        target = source.parent/'scene.xml'
        require_space(target, 1024**2)
        with target.open('xb') as stream:
            stream.write(ET.tostring(tree.getroot(), encoding='utf-8'))
    for name, digest in p['source_sha256'].items():
        frozen = output/'frozen-source'/name
        assert sha(frozen if frozen.exists() else ROOT/name) == digest, name
    records = []; frames = 0
    for row in summary['rows']:
        folder = output/row['case']; r = read(folder/'report.json')
        assert sha(folder/'report.json') == row['report_sha256'] and r['status'] == row['status']
        assert r['status'] != 'harness_error' and r['protocol_sha256'] == sha(protocol)
        original = ROOT/p['parent_evidence']/(row['case']+'-teacher')
        old = read(original/'report.json')
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
        with np.load(folder/'states.npz', allow_pickle=False) as z:
            state = {k: z[k] for k in z.files}
        with np.load(original/'states.npz', allow_pickle=False) as z:
            assert np.array_equal(z['qpos'][0], state['qpos'][0]) and np.array_equal(z['qvel'][0], state['qvel'][0])
        n = len(state['time']); frames += n
        assert n == r['trace_frames'] and np.all(np.diff(state['time']) > 0)
        assert state['qpos'].shape == (n, model.nq) and state['qvel'].shape == (n, model.nv)
        assert all(np.isfinite(state[k]).all() for k in ('qpos', 'qvel', 'targets', 'time'))
        assert r['setup']['valid'] == old['setup']['valid']
        assert all(r[k] == 0 for k in ('physics_state_writes_during_control', 'hidden_forces', 'equality_constraints'))
        metric = r['monitor']['metrics']
        data.qpos[:] = state['qpos'][-1]; data.qvel[:] = state['qvel'][-1]
        mujoco.mj_forward(model, data)
        if r['status'] in ('succeeded', 'already_open'):
            assert data.joint('drawer_slide').qpos[0] >= .105
            assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
            assert np.max(np.abs(data.qvel[:12])) < .12
            assert not r['monitor']['failure'] and metric['unexpected_collisions'] == 0
            assert metric['other_object_max_displacement_m'] <= .004 and metric['parked_arm_max_motion_deg'] <= 1.
            if r['status'] == 'succeeded':
                assert r['monitor']['passed'] and metric['verified_handle_contact_s'] >= .1
                assert metric['stable_release_and_park_s'] >= .5
            else:
                assert r['no_op'] and r['noop_stable_seconds'] >= .5
                assert metric['verified_handle_contact_s'] == 0.
            require_space(folder/'verification-views.png', 8*1024**2)
            model.vis.quality.offsamples = 0
            renderer = mujoco.Renderer(model, width=640, height=360)
            option = mujoco.MjvOption(); option.geomgroup[3:] = 0
            images = []
            try:
                for index in (0, n//2, n-1):
                    data.qpos[:] = state['qpos'][index]; data.qvel[:] = state['qvel'][index]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera='opposite', scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    rgb = renderer.render().copy()
                    assert rgb.mean() > 10
                    images.append(rgb)
            finally:
                renderer.close()
            with (folder/'verification-views.png').open('xb') as stream:
                Image.fromarray(np.concatenate(images, axis=1)).save(stream, format='PNG')
        records.append({**row, 'old_teacher_status': old['status'], 'trace_frames': n,
                        'unchanged_initial_state_verified': True})
    audit = {'schema': p['schema'], 'all_15_outcomes_retained': True, 'rows': records,
             'trace_frames': frames, 'physical_successes': sum(r['status'] == 'succeeded' for r in records),
             'old_teacher_successes': sum(r['old_teacher_status'] == 'succeeded' for r in records),
             'already_open': sum(r['status'] == 'already_open' for r in records),
             'source_copies': copied, 'preflight': preflight, 'promotion': False,
             'verification': 'All hashes, matching original qpos/qvel, finite state traces, terminal opening/parking geometry and every reported success threshold checked. Successful and semantic no-op initial/mid/final RGB views rendered from saved states. 20 Hz traces do not reconstruct every 200 Hz contact query.'}
    put(output/'audit.json', audit)
    copy(Path(__file__), output/'audit-source.py')
    text = f'''# Drawer configuration feasibility diagnosis

All **15 exposed starts** are retained and exactly match the original diagnostic. The cabinet-aligned, remaining-stroke programmed controller completes **5 physical openings**, compared with **2** for the original controller. This includes the anchor and its duplicate standard cabinet case, plus three recovered configurations: cabinet X=-0.24 m, cabinet yaw=+0.26 rad, and initial opening=0.04 m.

The initial 0.12 m opening passes a separate **already-open semantic no-op** check. It is not counted as a physical opening. The 0.08 m start and remaining displaced/rotated configurations still fail planning. A failed planner is not proof that a configuration is unreachable.

This teacher uses exact simulator state and inverse kinematics to produce physical-contact demonstrations. It is **not learned control** and is not installed in the selected browser. Its only changes rotate grasp/approach/pull geometry with the cabinet and pull the remaining distance to 0.116 m. Original torque, collision, contact, release and parking checks remain in force; there is no drawer actuator, state overwrite, attachment or hidden force.

The independent audit verifies all outcomes and **{frames:,}** saved state frames. Every failure, source, portable scene and original comparison hash remains available. A separate compact demonstration, image-observer and learned-motion experiment is needed before any new learned drawer claim.
'''
    target = output/'README.md'; require_space(target, 1024**2)
    with target.open('xb') as stream:
        stream.write(text.encode())
    put(output/'manifest.json', {'schema': p['schema'], 'files': {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}})
    print({k: audit[k] for k in ('trace_frames', 'physical_successes', 'old_teacher_successes', 'already_open')}, flush=True)


if __name__ == '__main__':
    run()

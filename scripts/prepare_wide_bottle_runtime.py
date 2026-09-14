"""Package the passing wider-bottle model and compare its task wrapper to evidence."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from scripts.bottle_refinement_experiment import read, sha, space, write, arrays
from scripts.evaluate_refined_bottle_physics import setup
from simulation_lab.wide_bottle_task import WideBottleTask, load_profile

PROTOCOL = ROOT/'docs/robotics/experiments/bottle-wide-physical-v1.json'
DEPLOYMENT = ROOT/'docs/robotics/experiments/bottle-wide-deployment-v1.json'


def prepare():
    p = read(PROTOCOL); evidence = ROOT/p['evidence_package']
    audit = read(evidence/'audit.json')
    assert audit['gate_passed'] and audit['all_118_outcomes_retained']
    assert all(all(v.values()) for v in audit['requirements'].values())
    for name, digest in read(evidence/'manifest.json')['files'].items():
        assert sha(evidence/name) == digest, name
    spec = read(evidence/'physical/integration.json')
    for name, digest in spec['frozen_sha256'].items():
        assert sha(ROOT/name) == digest, name
    output = ROOT/p['model_package']; raw = ROOT/'.run/bottle-wide-deployment-v1'
    if output.exists() or raw.exists() or DEPLOYMENT.exists():
        raise FileExistsError('Preserve runtime packages and deployment declarations.')
    required = sum(f.stat().st_size for key in ('observer', 'motor') for f in (ROOT/spec[key]).rglob('*') if f.is_file())
    preflight = space(p, output, required+32*1024**2)
    copied = {}
    for key in ('observer', 'motor'):
        source = ROOT/spec[key]
        for file in source.rglob('*'):
            if file.is_file():
                target = output/key/file.relative_to(source)
                space(p, target, file.stat().st_size+1024**2)
                write(target, file.read_bytes())
                copied[target.relative_to(ROOT).as_posix()] = {'source': file.relative_to(ROOT).as_posix(), 'sha256': sha(file)}
    # Entry-point/UI files receive a separate production test. The reusable
    # camera, neural-control and physical-monitor dependencies remain bound.
    excluded = {'engine.py', 'server.py', 'public_trial.py', 'cli.py'}
    sources = {name: digest for name, digest in spec['frozen_sha256'].items()
               if name.startswith('simulation_lab/') and (not name.endswith('.py') or Path(name).name not in excluded)}
    sources['simulation_lab/wide_bottle_task.py'] = sha(ROOT/'simulation_lab/wide_bottle_task.py')
    sources.update({name: item['sha256'] for name, item in copied.items()})
    evidence_files = [evidence/'audit.json', evidence/'manifest.json', PROTOCOL,
                      ROOT/spec['routing'], ROOT/p['camera_protocol'],
                      ROOT/'docs/robotics/evidence/bottle-refinement-v1/fresh-perception.json']
    sources.update({f.relative_to(ROOT).as_posix(): sha(f) for f in evidence_files})
    profile = {'schema': 'talos.bottle-wide-profile.v1', 'name': 'Wide bottle · live vision',
        'observer': (output/'observer').relative_to(ROOT).as_posix(),
        'motor': (output/'motor').relative_to(ROOT).as_posix(), 'routing': spec['routing'],
        'camera_protocol': p['camera_protocol'], 'camera_configuration': p['camera_configuration'],
        'physical_audit': (evidence/'audit.json').relative_to(ROOT).as_posix(), 'bound_files': sources,
        'source_workspace_m': p['physical']['workspace_m'], 'source_observation_margin_m': .005,
        'margin_interpretation': 'A 5 mm image-localization tolerance for the source-region refusal check, not an extension of the tested physical source range.',
        'destinations_xy_m': p['physical']['destinations_xy_m'],
        'standalone_perception_gate_remains_failed': True,
        'scope': spec['scope'], 'production_integration_verified': False,
        'model_copies': copied, 'preflight': preflight}
    write(output/'profile.json', profile)
    write(output/'README.md', b'''# Wider bottle controller V1

This package copies the unchanged RGB observer and V5 neural motor that passed the separate wider-bottle physical protocol. `profile.json` binds the models, reusable runtime, two destinations and complete physical audit. Exact fitting inputs and all failed predecessor experiments remain in their original published packages.

The standalone synthetic perception protocol remains failed. Live correction covers approach/descent and the closure-time offset; transport then uses motor feedback. This is not arbitrary workspace coverage, continuous visual carry correction or an airborne handoff.

The profile is prepared for separate task-wrapper and production tests. It is not selected in the browser merely because these files exist. Final deployment evidence must be reviewed before the new mode is exposed.
''')
    files = {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}
    write(output/'manifest.json', {'schema': profile['schema'], 'files': files})
    declaration = {'schema': 'talos.bottle-wide-deployment.v1', 'declared_on': '2026-09-14',
        'raw_root': raw.relative_to(ROOT).as_posix(), 'profile': (output/'profile.json').relative_to(ROOT).as_posix(),
        'physical_protocol': PROTOCOL.relative_to(ROOT).as_posix(),
        'physical_protocol_sha256': sha(PROTOCOL), 'profile_sha256': sha(output/'profile.json'),
        'source_sha256': {'scripts/prepare_wide_bottle_runtime.py': sha(Path(__file__)),
                          'simulation_lab/wide_bottle_task.py': sha(ROOT/'simulation_lab/wide_bottle_task.py')},
        'parity_seeds': p['physical']['evaluation_seeds'][:6], 'mode': 'live',
        'parity_gate': {'required_passes': 6, 'maximum_absolute_qpos_qvel_target_difference': 1e-9,
                        'require_same_frame_count_and_times': True, 'require_same_physical_status': True},
        'maximum_simulated_seconds': 100., 'maximum_wall_seconds_per_case': 180.,
        'scope': 'Exposed implementation regressions on the first six final scenes. They do not increase the fresh-sample denominator. Compare all recorded numeric physical/control arrays against the frozen evaluator before adding app entry points.',
        'stop': 'Preserve all six outcomes. On any mismatch do not expose the mode; diagnose separately without rewriting this gate. Fresh app/hosted entry-point checks require a separately frozen test declaration after wrapper parity passes.'}
    write(DEPLOYMENT, declaration)
    load_profile()
    print({'model_files': len(files), 'bound_dependencies': len(sources), 'profile_sha256': sha(output/'profile.json'),
           'parity_protocol_sha256': sha(DEPLOYMENT)}, flush=True)


def parity():
    p = read(PROTOCOL); d = read(DEPLOYMENT); root = ROOT/d['raw_root']/'parity'
    if root.exists():
        raise FileExistsError('Preserve all previous wrapper regressions.')
    assert sha(PROTOCOL) == d['physical_protocol_sha256'] and sha(ROOT/d['profile']) == d['profile_sha256']
    for name, digest in d['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    space(p, root, 256*1024**2); root.mkdir(parents=True)
    write(root/'protocol.json', d)
    evidence = ROOT/p['evidence_package']; spec = read(evidence/'physical/integration.json')
    torch.set_num_threads(2)
    rows = []
    for seed in d['parity_seeds']:
        folder = root/str(seed)
        if folder.exists():
            raise FileExistsError('Preserve a preceding wrapper rollout.')
        preflight = space(p, folder, 24*1024**2); folder.mkdir()
        trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
        task = None; started = time.perf_counter()
        report = {'seed': seed, 'passed': False, 'preflight': preflight,
                  'deployment_protocol_sha256': sha(DEPLOYMENT)}
        try:
            model, data, layout, destination, state = setup(p, spec, seed, 'evaluation', folder)
            report.update(setup=state, destination_m=destination, layout=layout)
            task = WideBottleTask(model, data, layout, destination)
            target = data.ctrl.copy()
            for tick in range(int(d['maximum_simulated_seconds']/model.opt.timestep)+1):
                q, v = data.qpos.copy(), data.qvel.copy()
                task.update(target)
                if not np.array_equal(q, data.qpos) or not np.array_equal(v, data.qvel):
                    raise ValueError('Wrapper changed authoritative state.')
                if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise ValueError('Hidden forces or constraints.')
                if tick % 10 == 0 or not task.active:
                    for key, value in zip(trace, (q, v, float(data.time), task.stage, target.copy())):
                        trace[key].append(value)
                if not task.active:
                    break
                if time.perf_counter()-started > d['maximum_wall_seconds_per_case']:
                    raise TimeoutError('Wrapper regression wall-time limit.')
                if tick % 2000 == 0:
                    space(p, folder, 24*1024**2)
                data.ctrl[:] = task.apply_gripper_limit(target)
                mujoco.mj_step(model, data)
            original_folder = evidence/'physical/evaluation'/f'{seed}-live'
            original = read(original_folder/'report.json')
            differences = {}; requirements = {'physical_status': task.status == original['status'] == 'succeeded'}
            with np.load(original_folder/'states.npz', allow_pickle=False) as z:
                for key in ('qpos', 'qvel', 'time', 'targets'):
                    actual = np.asarray(trace[key]); expected = z[key]
                    requirements[key+'_shape'] = actual.shape == expected.shape
                    difference = float(np.max(np.abs(actual-expected))) if actual.shape == expected.shape else None
                    differences[key] = difference
                    requirements[key+'_values'] = difference is not None and difference <= (0. if key == 'time' else 1e-9)
            report.update(status=task.status, task=task.snapshot(), completed_legs=task.completed_legs,
                physical=task.final_physical, requirements=requirements, maximum_absolute_differences=differences,
                passed=bool(all(requirements.values())), physics_state_writes_during_control=0,
                hidden_forces=0, inverse_solver_calls_during_control=0)
        except Exception as exc:
            report.update(status='harness_error', error_type=type(exc).__name__, message=str(exc))
        finally:
            if task is not None:
                task.close()
                write(folder/'observations.json', task.observation_log)
            arrays(folder/'states.npz', **{k: np.asarray(v) for k, v in trace.items()})
            report.update(wall_seconds=time.perf_counter()-started, trace_frames=len(trace['time']))
            write(folder/'report.json', report)
        row = {k: report.get(k) for k in ('seed', 'status', 'passed', 'message', 'maximum_absolute_differences')}
        row['report_sha256'] = sha(folder/'report.json'); rows.append(row)
        print(row, flush=True)
    passed = len(rows) == 6 and all(r['passed'] for r in rows)
    write(root/'gate.json', {'schema': d['schema'], 'deployment_protocol_sha256': sha(DEPLOYMENT),
                            'passed': passed, 'rows': rows, 'all_six_outcomes_retained': True})
    print({'wrapper_parity_passed': passed}, flush=True)
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'parity'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        raise SystemExit(parity())

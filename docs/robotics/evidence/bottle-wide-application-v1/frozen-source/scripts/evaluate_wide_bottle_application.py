"""Frozen app-entry and fresh full-dinner tests for wider bottle control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.engine import LabEngine
from simulation_lab import mug_visual_profile, public_trial, wide_bottle_task
from simulation_lab.storage import require_space
from scripts.measure_manipulation_coverage import read, sha, put
from scripts.evaluate_visual_mug_deployment import scene_copy
from scripts.package_spoon_release import physical_criteria
from scripts.package_bottle_wide import check_physical


def declare(path):
    parity = ROOT/'docs/robotics/evidence/bottle-wide-deployment-v1/parity/audit.json'
    assert read(parity)['passed'] and read(parity)['all_six_outcomes_retained']
    profile = wide_bottle_task.load_profile(); mug_visual_profile.load_profile()
    entry = []
    for surface in ('engine', 'public'):
        for name, seed, preset, instruction in (
            ('bottle-default', 42, 'wide_rectangle', 'Place. The bottle.'),
            ('bottle-right', 2026113001, 'wide_rectangle', 'Place the bottle at the right spot'),
            ('table-wide', 42, 'wide_rectangle', 'Set the table'),
            ('table-upright', 42, 'upright', 'Set the table')):
            entry.append({'id': surface+'-'+name, 'surface': surface, 'seed': seed,
                          'preset': preset, 'instruction': instruction, 'mode': 'candidate'})
    sources = [Path(__file__), ROOT/'scripts/evaluate_visual_mug_deployment.py',
               ROOT/'scripts/package_spoon_release.py', ROOT/'scripts/package_bottle_wide.py',
               ROOT/'scripts/measure_manipulation_coverage.py', *(ROOT/'simulation_lab').glob('*.py'),
               ROOT/'simulation_lab/web/app.js', ROOT/'simulation_lab/web/index.html', ROOT/'hosting/gradio_app.py']
    p = {'schema': 'talos.bottle-wide-application.v1', 'declared_on': '2026-09-14',
         'raw_root': '.run/bottle-wide-application-v1', 'entry_cases': entry,
         'development_seeds': list(range(2026113101, 2026113105)),
         'evaluation_seeds': list(range(2026113201, 2026113211)),
         'paired_modes': ['candidate', 'baseline'], 'maximum_workers': 2,
         'maximum_simulated_seconds': 420., 'maximum_wall_seconds': 420.,
         'storage_budget_bytes': 8*1024**3, 'reserve_gib': 10,
         'source_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sources},
         'profile_sha256': sha(wide_bottle_task.DEFAULT_PROFILE),
         'mug_profile_sha256': sha(mug_visual_profile.DEFAULT_PROFILE),
         'parity_audit_sha256': sha(parity),
         'entry_gate': 'All eight real local/public entry-point cases must complete physically before fresh full-workflow tests.',
         'development_gate': {'minimum_candidate_full_successes': 3, 'minimum_candidate_bottle_successes': 4,
                              'require_candidate_more_successes_than_baseline': True},
         'evaluation_gate': {'minimum_candidate_full_successes': 8, 'minimum_candidate_bottle_successes': 9,
                             'minimum_more_full_successes_than_baseline': 5},
         'source_distribution': 'The same wider uniform bottle X/Y/yaw reset recipe and original seeded mass/friction/light/other-object jitter. Reject >1 mm initial bottle overlap before settling, preserve that refusal in the planned denominator, and never resample. All other objects use their existing task-preset distribution.',
         'controls': 'Candidate uses the unchanged copied wide-bottle wrapper, then the existing plate/mug/drawer/fork/spoon skills with live late-mug correction. Baseline uses the preserved live-mug dinner profile and its original camera planner. Both receive the same full-table instruction and identically seeded resets. The candidate bottle destination is the tested [0.10,-0.115]; the baseline retains its original target semantics.',
         'scoring': 'Keep unchanged per-skill physical contact, hold, release, parked-arm, collision and disturbance criteria. Each completed bottle relay leg must pass. Verify all six requested skills and final sequence result. Refusal or invalid reset is a planned failure; distinguish it from a harness error.',
         'recording': 'Retain compact 20 Hz state/control traces, all outcomes, current RGB observation records, source/scene files and public initial/final views. Engine/public targets are the actual torque-limited position commands at recorded steps.',
         'scope': 'Entry cases are implementation regressions. Fresh full workflows test wider bottle starts with other objects still in narrow trained regions. They do not establish arbitrary arrangements, varied cabinet poses, other-object destination coverage or continuous visual carry correction.',
         'stop': 'Stop on an entry/development gate failure or harness error; later declared stages stay unexposed. Do not change weights, controller, thresholds, source or seeds after declaration. Any repair needs a separately declared protocol and preserved outcomes.'}
    if path.exists() or (ROOT/p['raw_root']).exists():
        raise FileExistsError('Preserve the app experiment and outputs.')
    p['preflight'] = require_space(ROOT/p['raw_root'], p['storage_budget_bytes'])
    put(path, p)
    print({'entry_cases': 8, 'development_rollouts': 8, 'final_rollouts': 20, 'protocol_sha256': sha(path)}, flush=True)


def cases(p, stage):
    if stage == 'entry':
        return p['entry_cases']
    return [{'id': f'{seed}-{mode}', 'surface': 'engine', 'seed': seed,
             'preset': 'wide_rectangle', 'instruction': 'Set the table', 'mode': mode}
            for seed in p[stage+'_seeds'] for mode in p['paired_modes']]


def check_frozen(p, path):
    for name, digest in p['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Declared application source changed: '+name)
    assert sha(wide_bottle_task.DEFAULT_PROFILE) == p['profile_sha256']
    assert sha(mug_visual_profile.DEFAULT_PROFILE) == p['mug_profile_sha256']


def run_case(args):
    p = read(args.protocol); check_frozen(p, args.protocol)
    case = next(c for c in cases(p, args.stage) if c['id'] == args.case)
    folder = ROOT/p['raw_root']/args.stage/case['id']
    if folder.exists():
        raise FileExistsError('Preserve every app rollout.')
    preflight = require_space(folder, 64*1024**2); folder.mkdir(parents=True)
    torch.set_num_threads(2)
    trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'control_stage', 'targets')}
    holders, initial_states, wide_children = [], [], []
    engine = None; snapshot = None; updates = []
    started = time.perf_counter()
    result = {'case': case, 'stage': args.stage, 'protocol_sha256': sha(args.protocol), 'preflight': preflight,
              'status': 'failed', 'passed': False, 'bottle_completed': False}
    original_wide_init = wide_bottle_task.WideBottleTask.__init__
    def retain_wide_child(task, *values, **kwargs):
        original_wide_init(task, *values, **kwargs)
        wide_children.append(task)
    tracker = patch.object(wide_bottle_task.WideBottleTask, '__init__', retain_wide_child)
    tracker.start()
    def capture(model, data, task, targets):
        if trace['time'] and trace['time'][-1] == float(data.time):
            return
        if len(trace['time']) % 200 == 0:
            require_space(folder, 32*1024**2)
        child = getattr(task, 'child', None) or task
        for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time),
                                     getattr(child, 'skill', 'planning'), child.stage, targets.copy())):
            trace[key].append(value)
    try:
        if case['surface'] == 'engine':
            engine = LabEngine(width=320, height=240)
            engine._reset(seed=case['seed'], scenario='dinner', dinner_preset='task', bottle_start=case['preset'])
            scene_copy(engine.xml, folder)
            initial_states.append((engine.data.qpos.copy(), engine.data.qvel.copy()))
            result['layout'] = engine.layout
            try:
                engine._language_command({'text': case['instruction'], 'mode': 'learned_dinner_wide' if case['mode'] == 'candidate' else 'learned_dinner_visual'})
            except ValueError as exc:
                result.update(status='refused', message=str(exc))
                capture(engine.model, engine.data, engine.task, engine.target)
            else:
                holders.append(engine.task)
                for tick in range(int(p['maximum_simulated_seconds']/engine.model.opt.timestep)+1):
                    q, v = engine.data.qpos.copy(), engine.data.qvel.copy()
                    engine.task.update(engine.target)
                    if not np.array_equal(q, engine.data.qpos) or not np.array_equal(v, engine.data.qvel):
                        raise RuntimeError('Task wrote authoritative state.')
                    if engine.model.neq or np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied):
                        raise RuntimeError('Hidden forces or constraints.')
                    applied = engine.task.apply_gripper_limit(engine.target)
                    if tick % 10 == 0 or not engine.task.active:
                        capture(engine.model, engine.data, engine.task, applied)
                    if not engine.task.active:
                        break
                    if time.perf_counter()-started > p['maximum_wall_seconds']:
                        engine.task.cancel(engine.target); result['time_limit'] = True; break
                    engine.data.ctrl[:] = applied
                    mujoco.mj_step(engine.model, engine.data)
                if engine.task.active:
                    engine.task.cancel(engine.target)
                snapshot = engine.task.snapshot()
        else:
            module = wide_bottle_task if case['mode'] == 'candidate' else mug_visual_profile
            original_make, original_build, original_step = module.make_sequence, public_trial.build_scene, mujoco.mj_step
            ticks = [0]
            def observe_make(model, data, layout, *values, **kwargs):
                initial_states.append((data.qpos.copy(), data.qvel.copy())); result['layout'] = layout
                task = original_make(model, data, layout, *values, **kwargs); holders.append(task); return task
            def observe_build(*values, **kwargs):
                xml, layout = original_build(*values, **kwargs); scene_copy(xml, folder); return xml, layout
            def observe_step(model, data, *values, **kwargs):
                if holders:
                    if ticks[0] % 10 == 0:
                        capture(model, data, holders[0], data.ctrl)
                    ticks[0] += 1
                return original_step(model, data, *values, **kwargs)
            first_image = final_image = None
            with patch.object(module, 'make_sequence', observe_make), patch.object(public_trial, 'build_scene', observe_build), patch.object(mujoco, 'mj_step', observe_step):
                for rgb, message, report in public_trial.run_trial(case['instruction'], seed=case['seed'],
                    controller='learned_wide' if case['mode'] == 'candidate' else 'learned_visual',
                    bottle_start=case['preset'], camera='opposite', cache_dir=folder, max_wall_seconds=p['maximum_wall_seconds']):
                    updates.append({'message': message, 'report': report})
                    if rgb is not None:
                        if first_image is None:first_image = rgb.copy()
                        final_image = rgb.copy()
                if holders:
                    task = holders[0]; capture(task.model, task.data, task, task.data.ctrl); snapshot = task.snapshot()
            put(folder/'public-updates.json', updates)
            for name, rgb in (('initial', first_image), ('final', final_image)):
                if rgb is not None:
                    if rgb.mean() < 10:raise RuntimeError('Public preview is black.')
                    require_space(folder/(name+'.png'), 1024**2)
                    with (folder/(name+'.png')).open('xb') as stream:Image.fromarray(rgb).save(stream, format='PNG')
            result['public_report'] = updates[-1]['report'] if updates else None
            if not holders:
                result.update(status='refused', message=updates[-1]['message'] if updates else 'No public task was constructed.')
        if snapshot is not None:
            result.update(status=snapshot['status'], message=snapshot['message'], task=snapshot)
            full = case['instruction'] == 'Set the table'
            expected = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon'] if full else ['bottle']
            succeeded = snapshot['status'] == 'succeeded' and snapshot.get('completed_steps') == expected
            if case['mode'] == 'baseline':
                # The preserved camera planner may insert its two reverse-relay
                # legs. Its unchanged goal and completion semantics are retained.
                succeeded = snapshot['status'] == 'succeeded' and all(s in snapshot.get('completed_steps', []) for s in ('plate','mug','drawer','fork','spoon'))
            rows = snapshot.get('results', [])
            for row in rows:
                if not full and case['mode'] == 'candidate':continue
                if row['status'] != 'succeeded':continue
                if row.get('policy_mode') == 'learned_bottle_wide':
                    assert row['teacher_updates'] == 0 and row['inverse_solver_calls_during_control'] == 0
                    check_physical({'passed': True, 'failure': None, 'metrics': row['metrics']})
                else:
                    physical_criteria(row)
            result['bottle_completed'] = any(row.get('skill') == 'bottle' and row['status'] == 'succeeded' for row in rows)
            if not full and case['mode'] == 'candidate':
                result['bottle_completed'] = snapshot['status'] == 'succeeded'
                for leg in holders[0].completed_legs:check_physical(leg['physical'])
                result['completed_legs'] = holders[0].completed_legs
            if full and case['mode'] == 'candidate':
                mug = next((r for r in rows if r.get('skill') == 'mug'), None)
                if mug and mug['status'] == 'succeeded':
                    assert mug['mug_correction_mode'] == 'live' and mug['mug_visual_corrections']
            for child in wide_children:
                for leg in child.completed_legs:check_physical(leg['physical'])
            result.update(passed=bool(succeeded), physics_state_writes_during_control=0, hidden_forces=0, equality_constraints=0)
    except ValueError as exc:
        result.update(status='refused', message=str(exc))
    except Exception as exc:
        result.update(status='harness_error', error_type=type(exc).__name__, message=str(exc))
        if snapshot is not None:result['task'] = snapshot
    finally:
        tracker.stop()
        if wide_children:
            put(folder/'bottle-runtime.json', {'tasks': [{'snapshot': child.snapshot(), 'completed_legs': child.completed_legs,
                'observations': child.observation_log} for child in wide_children]})
        if holders and isinstance(holders[0], wide_bottle_task.WideBottleTask):
            put(folder/'observations.json', holders[0].observation_log)
        if initial_states:
            require_space(folder/'initial.npz', 1024**2)
            with (folder/'initial.npz').open('xb') as stream:np.savez_compressed(stream, qpos=initial_states[0][0], qvel=initial_states[0][1])
        require_space(folder/'states.npz', 32*1024**2)
        with (folder/'states.npz').open('xb') as stream:np.savez_compressed(stream, **{k: np.asarray(v) for k, v in trace.items()})
        result.update(wall_seconds=time.perf_counter()-started, trace_frames=len(trace['time']))
        put(folder/'report.json', result)
        if engine is not None:
            if hasattr(engine, 'task') and hasattr(engine.task, 'close'):engine.task.close()
            engine.close()
    print({k: result.get(k) for k in ('case','status','passed','bottle_completed','message','wall_seconds','trace_frames')}, flush=True)
    return int(result['status'] == 'harness_error')


def batch(args):
    p = read(args.protocol); check_frozen(p, args.protocol); raw = ROOT/p['raw_root']
    folder = raw/args.stage
    if folder.exists():raise FileExistsError('Preserve preceding app stages.')
    for prior in ([] if args.stage == 'entry' else ['entry'] if args.stage == 'development' else ['entry', 'development']):
        gate = read(raw/prior/'gate.json')
        if not gate['passed'] or gate['protocol_sha256'] != sha(args.protocol):raise ValueError('The preceding application gate must pass unchanged.')
    require_space(folder, p['storage_budget_bytes']); folder.mkdir(parents=True)
    if args.stage == 'entry':
        put(raw/'protocol.json', p)
        for name in p['source_sha256']:
            target=raw/'frozen-source'/name; require_space(target,(ROOT/name).stat().st_size+1024); target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    def launch(case):
        log=folder/(case['id']+'.log')
        if log.exists() or (folder/case['id']).exists():raise FileExistsError('Preserve each output and log.')
        require_space(log,64*1024**2)
        with log.open('xb') as stream:
            code=subprocess.run([sys.executable,'-I',str(Path(__file__)),'--protocol',str(args.protocol.resolve()),'--stage',args.stage,'--case',case['id']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=p['maximum_wall_seconds']+90).returncode
        path=folder/case['id']/'report.json';r=read(path) if path.exists() else {'status':'harness_error','passed':False}
        row={'case':case,'status':r['status'],'passed':r['passed'],'bottle_completed':r.get('bottle_completed',False),'message':r.get('message'),'exit_code':code,'report_sha256':sha(path) if path.exists() else None}
        print(row,flush=True);return row
    rows=[];planned=cases(p,args.stage)
    with ThreadPoolExecutor(max_workers=p['maximum_workers']) as pool:
        for offset in range(0,len(planned),2):
            group=list(pool.map(launch,planned[offset:offset+2]));rows+=group
            if any(r['status']=='harness_error' or r['exit_code'] for r in group):
                put(folder/'stopped.json',{'reason':'Harness error; preserve attempted outcomes and leave later cases unexposed.','rows':rows});return 1
    if args.stage=='entry':
        counts={'candidate':sum(r['passed'] for r in rows)};bottles=sum(r['bottle_completed'] for r in rows);passed=counts['candidate']==8
    else:
        counts={mode:sum(r['case']['mode']==mode and r['passed'] for r in rows) for mode in p['paired_modes']}
        bottles=sum(r['case']['mode']=='candidate' and r['bottle_completed'] for r in rows)
        limits=p[args.stage+'_gate'];passed=counts['candidate']>=limits['minimum_candidate_full_successes'] and bottles>=limits['minimum_candidate_bottle_successes']
        passed=passed and (counts['candidate']>counts['baseline'] if args.stage=='development' else counts['candidate']>=counts['baseline']+limits['minimum_more_full_successes_than_baseline'])
    put(folder/'gate.json',{'schema':p['schema'],'protocol_sha256':sha(args.protocol),'passed':bool(passed),'counts':counts,'candidate_bottle_successes':bottles,'rows':rows,'all_planned_outcomes_retained':len(rows)==len(planned)})
    print({'stage':args.stage,'passed':bool(passed),'counts':counts,'bottles':bottles},flush=True)
    return int(not passed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',type=Path);parser.add_argument('--protocol',type=Path)
    parser.add_argument('--stage',choices=['entry','development','evaluation']);parser.add_argument('--case')
    args=parser.parse_args()
    if args.declare:declare(args.declare)
    elif args.protocol and args.stage:raise SystemExit(run_case(args) if args.case else batch(args))
    else:parser.error('Declare a protocol or choose a stage.')

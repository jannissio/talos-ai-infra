"""Freeze and compare complete browser-engine workflows for a declared release fit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from simulation_lab.engine import LabEngine
from simulation_lab.storage import GIB, require_space

PROTOCOL = None
BASE = None
PRESETS = ('upright','wide_left')
CONTROLLERS = ('baseline','candidate')
STEPS = {'upright':['bottle','plate','mug','drawer','fork','spoon'],
         'wide_left':['reverse_bottle_right','reverse_bottle_left','plate','mug','drawer','fork','spoon']}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def configure(protocol_path):
    global PROTOCOL, BASE
    PROTOCOL = Path(protocol_path).resolve()
    BASE = (ROOT/read(PROTOCOL)['raw_root']).resolve()
    if not BASE.is_relative_to(ROOT):
        raise ValueError('The declared experiment root must stay inside the repository.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def write(path,value):
    require_space(path,4*1024**2)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2)+'\n').encode())


def budget(path, expected):
    protocol = read(PROTOCOL)
    folders = (BASE,ROOT/protocol['training_package'],ROOT/protocol['model_package'],ROOT/protocol['evidence_package'])
    used = sum(p.stat().st_size for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file())
    used += sum(p.stat().st_size for p in (ROOT/'.run/final-goal').glob(BASE.name+'*') if p.is_file())
    if used+expected > protocol['budget']['maximum_data_gib_including_packages']*GIB:
        raise ValueError('Declared raw-plus-packaged evidence budget would be exceeded.')
    require_space(path,expected)


def inputs():
    protocol = read(PROTOCOL)
    training = read(BASE/'fit/training.json')
    if (not training['training_completed'] or training['completed_steps'] != protocol['training']['steps']
            or training['settings'] != protocol['training']):
        raise ValueError('A completed declared fit is required.')
    replay = ROOT/protocol['training_package']/'input-replay.json'
    if (training['protocol_sha256'] != sha(PROTOCOL) or not read(replay)['passed']
            or training['training_gate_sha256'] != sha(replay)
            or training['source_sha256'] != read(replay)['source_sha256']):
        raise ValueError('Protocol or training data gate changed.')
    if (sha(ROOT/protocol['baseline']/'primitive.safetensors') != protocol['baseline_sha256']
            or training['warm_start_sha256'] != protocol['baseline_sha256']):
        raise ValueError('The declared baseline or warm start changed.')
    suites = {'baseline':ROOT/protocol['baseline_suite'],'candidate':BASE/'candidate-suite.json'}
    baseline = read(suites['baseline'])
    candidate = read(suites['candidate'])
    if set(candidate) != set(baseline):
        raise ValueError('Only the declared skill model may be replaced.')
    for skill in baseline:
        if skill != protocol['skill'] and (suites['baseline'].parent/baseline[skill]).resolve() != (suites['candidate'].parent/candidate[skill]).resolve():
            raise ValueError('Another model changed: '+skill)
    model = (suites['candidate'].parent/candidate[protocol['skill']]).resolve()
    if sha(model/'primitive.safetensors') != training['selection']['sha256']:
        raise ValueError('Candidate differs from the sole completed fit.')
    original_meta = read(ROOT/protocol['baseline']/'primitive.json')
    candidate_meta = read(model/'primitive.json')
    provenance_fields = {'source_sha256','action_preprocessing','arguments','normalization_contract',
                         'training_completed','training_protocol','warm_start_sha256'}
    if ({k:v for k,v in original_meta.items() if k not in provenance_fields}
            != {k:v for k,v in candidate_meta.items() if k not in provenance_fields}):
        raise ValueError('A normalization, gripper, camera or controller metadata field changed.')
    for name in ('visual.npz','talos_normalization.json'):
        if sha(model/name) != sha(ROOT/protocol['baseline']/name):
            raise ValueError('Original visual/normalization data changed.')
    export = read(model/'benchmark.json')
    if export['source_sha256'] != training['selection']['sha256'] or not any(r['device']=='CPU' and r.get('parity_passed') for r in export['devices']):
        raise ValueError('The selected export does not pass CPU parity.')
    sources = sorted((ROOT/'simulation_lab').glob('*.py'))+[Path(__file__).resolve()]
    result = {'protocol_sha256':sha(PROTOCOL),'training_report_sha256':sha(BASE/'fit/training.json'),
              'training_gate_sha256':sha(replay),
              'source_sha256':{relative(p):sha(p) for p in sources},
              'asset_sha256':{relative(p):sha(p) for p in sorted((ROOT/'simulation_lab/assets').rglob('*')) if p.is_file()},
              'suites':{}}
    for name,suite in suites.items():
        files = {relative(suite):sha(suite)}
        for value in read(suite).values():
            folder = (suite.parent/value).resolve()
            # Visual NPZ files are runtime inputs as well as weights and IR.
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix in ('.safetensors','.json','.xml','.bin','.npz'):
                    files[relative(path)] = sha(path)
        result['suites'][name] = {'path':relative(suite),'files_sha256':files}
    return result


def aggregate(rows,split):
    protocol = read(PROTOCOL)
    seeds = protocol[split+'_seeds']
    expected = {(s,p,c) for s in seeds for p in PRESETS for c in CONTROLLERS}
    actual = {(r['seed'],r['preset'],r['controller']) for r in rows}
    if len(actual) != len(rows) or not actual <= expected:
        raise ValueError('Duplicate or undeclared paired trial.')
    counts = {c:{p:sum(r['passed'] for r in rows if r['controller']==c and r['preset']==p) for p in PRESETS} for c in CONTROLLERS}
    skill_passes = {c:{p:sum(r.get('target_skill_status')=='succeeded' for r in rows if r['controller']==c and r['preset']==p) for p in PRESETS} for c in CONTROLLERS}
    complete = actual == expected
    paired = []
    for seed in seeds:
        for preset in PRESETS:
            pair = [r for r in rows if r['seed']==seed and r['preset']==preset]
            if len(pair) == 2:
                states = [r.get('initial_state_sha256') for r in pair]
                paired.append({'seed':seed,'preset':preset,'match':bool(all(states) and states[0]==states[1])})
    pairs_valid = len(paired)==len(seeds)*len(PRESETS) and all(p['match'] for p in paired)
    if split == 'development':
        gate = complete and pairs_valid and all(counts['candidate'][p] >= 5 and counts['candidate'][p] >= counts['baseline'][p]
                                and skill_passes['candidate'][p] >= skill_passes['baseline'][p] for p in PRESETS)
    else:
        gate = complete and pairs_valid and all(counts['candidate'][p] >= 10 for p in PRESETS) and sum(counts['candidate'].values()) >= sum(counts['baseline'].values())+2
    return {'complete':complete,'attempted':len(rows),'planned':len(expected),'passed_workflows':counts,
            'passed_target_skill_attempts':skill_passes,'paired_initial_states_match':pairs_valid,'pairs':paired,
            'gate_passed':gate,'rows':sorted(rows,key=lambda r:(r['seed'],r['preset'],r['controller']))}


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve the existing selection freeze.')
    protocol = read(PROTOCOL)
    current = inputs()
    result = {'schema':protocol['schema'],'split':args.split,'inputs':current,
              'seeds':protocol[args.split+'_seeds'],'presets':list(PRESETS),'controllers':list(CONTROLLERS),
              'instruction':'set the table','expected_steps':STEPS,'maximum_simulation_seconds_per_trial':450,
              'maximum_worker_wall_seconds':900,
              'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'scope':'Paired actual language/RGB-engine workflows; changed declared skill only, unchanged physical thresholds.'}
    if args.split == 'evaluation':
        if not args.development:
            raise ValueError('Final freeze needs completed paired development.')
        development = read(args.development/'summary.json')
        prior = read(args.development/'frozen-inputs.json')
        for row in development['rows']:
            if row.get('report'):
                report_path = args.development/row['report']
                report = read(report_path)
                passed = report['status']=='succeeded' and report.get('task',{}).get('completed_steps')==STEPS[row['preset']]
                if sha(report_path)!=row['report_sha256'] or passed!=row['passed'] or report.get('initial_state_sha256')!=row.get('initial_state_sha256'):
                    raise ValueError('A development report differs from its summary.')
            elif row['passed']:
                raise ValueError('A passed trial must retain its physical report.')
        derived = aggregate(development['rows'],'development')
        if not derived['gate_passed'] or prior['inputs'] != current:
            raise ValueError('Development gate failed or evaluated inputs changed.')
        result.update(development_gate_passed=True,development_summary_sha256=sha(args.development/'summary.json'),
                      development_freeze_sha256=sha(args.development/'frozen-inputs.json'))
    budget(args.output,4*1024**2)
    write(args.output,result)
    print(json.dumps({'freeze':relative(args.output),'split':args.split,'planned_trials':len(result['seeds'])*4,
                      'sha256':sha(args.output)}))


def validate(frozen,split=None):
    protocol = read(PROTOCOL)
    split = split or frozen['split']
    if frozen['split'] != split or frozen['inputs'] != inputs():
        raise ValueError('Runtime, model, export or input hashes differ from the freeze.')
    if frozen['seeds'] != protocol[split+'_seeds'] or frozen['presets'] != list(PRESETS) or frozen['controllers'] != list(CONTROLLERS):
        raise ValueError('The paired scene conditions changed.')
    if (frozen['instruction']!='set the table' or frozen['expected_steps']!=STEPS
            or frozen['maximum_simulation_seconds_per_trial']!=450 or frozen['maximum_worker_wall_seconds']!=900):
        raise ValueError('The frozen command, expected workflow or operational limits changed.')
    if split == 'evaluation' and not frozen.get('development_gate_passed'):
        raise ValueError('Final evaluation requires the passed development gate.')


def trial(args):
    frozen = read(args.freeze)
    validate(frozen)
    if args.seed not in frozen['seeds'] or args.preset not in PRESETS or args.controller not in CONTROLLERS:
        raise ValueError('Undeclared physical trial.')
    if args.output.exists():
        raise FileExistsError('Preserve every prior trial.')
    budget(args.output,12*1024**2)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    engine = LabEngine(width=320,height=240)
    engine.dinner_suite = ROOT/frozen['inputs']['suites'][args.controller]['path']
    trace = {key:[] for key in ('qpos','qvel','time','stage','targets','progress')}
    started = time.perf_counter()
    report = {'seed':args.seed,'preset':args.preset,'controller':args.controller,'split':frozen['split'],
              'freeze_sha256':sha(args.freeze),'instruction':frozen['instruction'],'status':'execution_error',
              'physics_state_writes_during_control':0,'hidden_forces':0,'expected_steps':STEPS[args.preset]}
    try:
        engine._reset(seed=args.seed,scenario='dinner',dinner_preset='task',drawer_open=False,bottle_start=args.preset)
        scene = ET.fromstring(engine.xml)
        compiler = scene.find('compiler')
        meshdir = Path(compiler.get('meshdir')).resolve()
        if not meshdir.is_relative_to(ROOT/'simulation_lab/assets'):
            raise ValueError('Unexpected scene asset path.')
        compiler.set('meshdir',os.path.relpath(meshdir,args.output).replace('\\','/'))
        (args.output/'scene.xml').write_bytes(ET.tostring(scene,encoding='utf-8'))
        initial = np.r_[engine.data.qpos,engine.data.qvel,engine.data.ctrl].astype(np.float64)
        np.save(args.output/'initial-state.npy',initial)
        report['initial_state_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
        engine._language_command({'text':frozen['instruction'],'mode':'learned_dinner'})
        report['visual_plan'] = engine.task.plan
        if engine.task.steps != STEPS[args.preset]:
            raise ValueError('The actual RGB/language plan differs from the declared workflow.')
        limit = int(frozen['maximum_simulation_seconds_per_trial']/engine.model.opt.timestep)
        for tick in range(limit):
            before,velocity = engine.data.qpos.copy(),engine.data.qvel.copy()
            engine.task.update(engine.target)
            if not np.array_equal(before,engine.data.qpos) or not np.array_equal(velocity,engine.data.qvel):
                report['physics_state_writes_during_control'] += 1
                raise RuntimeError('Controller wrote physical state.')
            if engine.model.neq or np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied):
                report['hidden_forces'] += 1
                raise RuntimeError('Unexpected constraint or applied force.')
            if tick%10 == 0 or not engine.task.active:
                child = engine.task.child
                values = (engine.data.qpos.copy(),engine.data.qvel.copy(),float(engine.data.time),
                          child.skill,engine.target.copy(),float(child.policy.progress))
                for name,value in zip(trace,values):
                    trace[name].append(value)
            if tick%2000 == 0:
                budget(args.output,12*1024**2)
            if not engine.task.active:
                break
            engine.data.ctrl[:] = engine.task.apply_gripper_limit(engine.target)
            mujoco.mj_step(engine.model,engine.data)
        if engine.task.active:
            engine.task.cancel(engine.target)
        report.update(status=engine.task.status,task=engine.task.snapshot())
    except ValueError as exc:
        report.update(status='refused',message=str(exc))
    except Exception as exc:
        report.update(status='execution_error',error_type=type(exc).__name__)
    finally:
        report.update(wall_seconds=time.perf_counter()-started,simulation_seconds=float(engine.data.time))
        budget(args.output,12*1024**2)
        np.savez_compressed(args.output/'states.npz',**{key:np.asarray(values) for key,values in trace.items()})
        write(args.output/'report.json',report)
        if getattr(engine,'task',None) and hasattr(engine.task,'close'):
            engine.task.close()
        engine.close()
    print(json.dumps({'seed':args.seed,'preset':args.preset,'controller':args.controller,'status':report['status'],
                      'completed_steps':report.get('task',{}).get('completed_steps'),'wall_seconds':report['wall_seconds']}),flush=True)
    return report['status'] == 'succeeded'


def batch(args):
    frozen = read(args.freeze)
    validate(frozen)
    if args.output.exists():
        raise FileExistsError('Preserve the previous paired batch.')
    if not 1 <= args.workers <= read(PROTOCOL)['budget']['maximum_physical_workers']:
        raise ValueError('Unexpected worker count.')
    jobs = [(seed,preset,controller) for seed in frozen['seeds'] for preset in PRESETS for controller in CONTROLLERS]
    budget(args.output,len(jobs)*12*1024**2)
    args.output.mkdir(parents=True)
    # Copy the exact already-frozen bytes; each trial names this same digest.
    (args.output/'frozen-inputs.json').write_bytes(args.freeze.read_bytes())
    for name in frozen['inputs']['source_sha256']:
        target = args.output/'source'/name
        require_space(target,(ROOT/name).stat().st_size+1024**2)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    rows,started = [],time.perf_counter()

    def one(seed,preset,controller):
        name = f'{seed}-{preset}-{controller}'
        folder,log = args.output/name,args.output/(name+'.log')
        if folder.exists() or log.exists():
            raise FileExistsError('Both trial output and log must be unused.')
        budget(folder,12*1024**2)
        command = [sys.executable,str(Path(__file__).resolve()),'--protocol',str(PROTOCOL),'trial','--freeze',str(args.freeze.resolve()),
                   '--seed',str(seed),'--preset',preset,'--controller',controller,'--output',str(folder)]
        try:
            with log.open('x',encoding='utf-8') as stream:
                process = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=frozen['maximum_worker_wall_seconds'])
        except subprocess.TimeoutExpired:
            return dict(seed=seed,preset=preset,controller=controller,passed=False,status='execution_timeout')
        path = folder/'report.json'
        if not path.exists():
            return dict(seed=seed,preset=preset,controller=controller,passed=False,status='execution_error',exit_code=process.returncode)
        report = read(path)
        target_skill = next((r for r in report.get('task',{}).get('results',[]) if r['skill']==read(PROTOCOL)['skill']),None)
        passed = (process.returncode==0 and report['status']=='succeeded'
                  and report.get('task',{}).get('completed_steps') == STEPS[preset]
                  and not report['physics_state_writes_during_control'] and not report['hidden_forces'])
        return {'seed':seed,'preset':preset,'controller':controller,'passed':passed,'status':report['status'],
                'target_skill_status':None if target_skill is None else target_skill['status'],
                'target_skill_placement_mm':None if target_skill is None else target_skill['metrics'].get('placement_error_mm'),
                'initial_state_sha256':report.get('initial_state_sha256'),'report':name+'/report.json',
                'report_sha256':sha(path),'wall_seconds':report['wall_seconds']}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job in as_completed([pool.submit(one,*parameters) for parameters in jobs]):
            row = job.result()
            rows.append(row)
            result = aggregate(rows,frozen['split'])
            result.update(split=frozen['split'],freeze_sha256=sha(args.freeze),wall_seconds=time.perf_counter()-started)
            write(args.output/'summary.json',result)
            print(json.dumps({'attempted':len(rows),'planned':len(jobs),**row}),flush=True)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--split',choices=('development','evaluation'),required=True)
    p.add_argument('--development',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p = commands.add_parser('trial')
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--preset',choices=PRESETS,required=True)
    p.add_argument('--controller',choices=CONTROLLERS,required=True)
    p.add_argument('--output',type=Path,required=True)
    p = commands.add_parser('batch')
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    configure(args.protocol)
    if args.command == 'freeze':
        freeze(args)
    elif args.command == 'trial':
        raise SystemExit(0 if trial(args) else 1)
    else:
        batch(args)

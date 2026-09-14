"""Fresh task-level validation of frozen wider bottle control, without refitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bottle_refinement_experiment import read, sha, space, write
from scripts.evaluate_refined_bottle_physics import execute_trial, integration_path, checked_integration
from simulation_lab.bottle_refinement_runtime import RefinedBottleObserver


def declare(path):
    parent_path = ROOT/'docs/robotics/experiments/bottle-refinement-exposed-physics-v2.json'
    parent = read(parent_path)
    evidence = ROOT/parent['evidence_package']
    audit = read(evidence/'audit.json')
    assert audit['all_58_outcomes_retained'] and audit['all_original_starts_verified']
    for name, digest in read(evidence/'manifest.json')['files'].items():
        assert sha(evidence/name) == digest, name
    if path.exists():
        raise FileExistsError('Preserve every declared experiment.')
    source_names = ['scripts/evaluate_bottle_wide.py', 'scripts/evaluate_refined_bottle_physics.py',
        'scripts/bottle_refinement_experiment.py']
    source_names += [file.relative_to(ROOT).as_posix() for file in (ROOT/'simulation_lab').glob('*.py')]
    observer = ROOT/'models/bottle_refinement_v1'
    motor = ROOT/'docs/robotics/evidence/rgb-servo-v5/physical/model'
    routing = ROOT/'docs/robotics/experiments/rgb-servo-routing-v2.json'
    inputs = [parent_path, evidence/'audit.json', evidence/'manifest.json', routing, ROOT/parent['camera_protocol'],
        ROOT/'docs/robotics/experiments/bottle-refinement-v1.json',
        ROOT/'docs/robotics/evidence/bottle-refinement-v1/fresh-perception.json',
        motor/'experiment.json', motor/'motor.safetensors', motor/'openvino/motor.xml', motor/'openvino/motor.bin']
    inputs += [file for file in (observer/'openvino').iterdir() if file.is_file()]
    inputs += [observer/'refiner.safetensors', observer/'coarse.safetensors', observer/'model.json']
    inputs += [file for file in (ROOT/'models/bottle_visual').rglob('*') if file.is_file()]
    inputs += [file for file in (ROOT/'simulation_lab/assets/so101').rglob('*') if file.is_file()]
    frozen = {name: sha(ROOT/name) for name in source_names}
    frozen.update({file.relative_to(ROOT).as_posix(): sha(file) for file in inputs})
    p = {'schema': 'talos.bottle-wide-physical.v1', 'declared_on': '2026-09-14',
        'raw_root': '.run/bottle-wide-physical-v1', 'model_package': 'models/bottle_wide_v1',
        'training_package': 'training/bottle_wide_v1', 'evidence_package': 'docs/robotics/evidence/bottle-wide-physical-v1',
        'camera_protocol': parent['camera_protocol'], 'camera_configuration': parent['camera_configuration'],
        'physical': {'workspace_m': {'x': [-.14,.16], 'y': [-.18,-.06], 'yaw': [-.6,.6]},
            'destinations_xy_m': [[.10,-.115],[.20,.05]],
            'development_seeds': list(range(2026112501,2026112509)),
            'evaluation_seeds': list(range(2026112601,2026112633))},
        'frozen_sha256': frozen, 'regression_seeds': list(range(2026091301,2026091307)),
        'development_modes': ['live','baseline'], 'evaluation_modes': ['live','frozen','baseline'],
        'budget': {'maximum_raw_and_packaged_gib': 4, 'maximum_physical_trials': 118, 'reserve_gib': 10},
        'choice': 'Freeze the complete observer/motor/controller after the 29-case training-exposed diagnostic. No new fit, checkpoint selection, camera change, motor-guard relaxation or controller tuning belongs to this protocol.',
        'known_limitation': 'Standalone bottle-refinement-v1 failed its fresh synthetic perception gate (94.956% present acceptance, 4.427 mm maximum error). That failed result remains unchanged, and its original physical seed sets stay unexposed. This distinct experiment tests complete physical execution with unchanged 8 mm placement/contact/release/parking criteria; it does not certify the stricter standalone perception limits.',
        'candidate_basis': audit['total'],
        'destinations': 'Two already physically demonstrated targets. The middle target [0.16,-0.06] failed the exposed route/motor diagnostic and is explicitly unsupported by any resulting profile. The source rectangle retains its corresponding corner; source refusals remain failures, not exclusions.',
        'randomization': 'Each declared scene seed produces a fresh dinner task scene: other-object XY up to 9 mm (cutlery 3 mm), mass +/-8%, friction +/-10%, light multiplier 0.92..1.06. Replace only bottle source XY/yaw with independent uniform draws from the declared bounds; settle 300 steps. Geometry, camera calibration, arm bases and closed drawer configuration remain fixed.',
        'comparison': 'All modes start from identical scene/physical state. Baseline is the unchanged preset policy and cannot condition on new requested destinations. Report each goal separately; live/frozen is the controlled image-feedback comparison.',
        'validity': 'No resampling. Invalid initial overlap/tipping/support/settling cases are retained in planned counts. Perception and neural-motor refusals remain valid-case failures. Failed solver/route checks do not prove a point unreachable.',
        'development_gate': {'minimum_live_successes': 6, 'minimum_live_per_destination': 3,
            'strictly_more_than_baseline': True, 'maximum_harness_errors': 0},
        'evaluation_gate': {'minimum_live_successes': 26, 'minimum_live_per_destination': 13,
            'minimum_live_fraction_of_valid': .9, 'minimum_more_successes_than_baseline': 12,
            'minimum_more_successes_than_frozen': 8, 'required_exposed_regression_successes': 6,
            'maximum_harness_errors': 0},
        'stop': 'Complete each declared stage, retaining all failures; stop immediately for a harness error. A failed development gate leaves every final/regression case unexposed. No threshold, camera, weight or route change after exposure. A passing gate allows only a separate guarded integration and production checks, not general all-object or all-position claims.'}
    raw = ROOT/p['raw_root']
    if raw.exists():
        raise FileExistsError('Preserve earlier wider-physical experiments.')
    preflight = space(p, raw, 1024**3)
    RefinedBottleObserver(observer/'openvino')
    write(path,p)
    spec = {'schema': p['schema'], 'protocol_sha256': sha(path), 'frozen_sha256': frozen,
        'observer': (observer/'openvino').relative_to(ROOT).as_posix(),
        'motor': motor.relative_to(ROOT).as_posix(), 'routing': routing.relative_to(ROOT).as_posix(),
        'baseline': 'models/bottle_visual', 'settle_steps': 300,
        'maximum_simulated_seconds': 100., 'maximum_wall_seconds': 180., 'maximum_workers': 2,
        'source_parent_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'preflight': preflight, 'scope': 'Frozen wider upright-bottle execution at two explicit destinations. Source/scene randomization is declared; no arbitrary task, unknown geometry, moved cabinet or continuous visual carry correction claim.'}
    write(integration_path(p),spec)
    for name in source_names:
        write(raw/'physical/frozen-source'/name,(ROOT/name).read_bytes())
    print({'protocol_sha256':sha(path),'development_trials':16,'final_comparisons_and_regressions':102},flush=True)


def check_final(p,protocol):
    raw=ROOT/p['raw_root']/'physical'
    gate=read(raw/'development/gate.json'); frozen=read(raw/'evaluation-freeze.json')
    if not gate['passed'] or frozen['development_gate_sha256']!=sha(raw/'development/gate.json') or frozen['integration_sha256']!=sha(integration_path(p)) or frozen['protocol_sha256']!=sha(protocol):
        raise ValueError('Final cases require a passing and unchanged frozen development gate.')


def trial(args):
    p=read(args.protocol); spec=checked_integration(p,args.protocol)
    seeds=p['regression_seeds'] if args.split=='regression' else p['physical'][args.split+'_seeds']
    modes=['live'] if args.split=='regression' else p[args.split+'_modes']
    if args.seed not in seeds or args.mode not in modes:
        raise ValueError('Undeclared trial condition.')
    if args.split in ('evaluation','regression'):
        check_final(p,args.protocol)
    folder=ROOT/p['raw_root']/'physical'/args.split/f'{args.seed}-{args.mode}'
    return execute_trial(p,spec,args,folder)


def batch(args):
    p=read(args.protocol); spec=checked_integration(p,args.protocol)
    raw=ROOT/p['raw_root']/'physical'
    splits=['development'] if args.split=='development' else ['evaluation','regression']
    if any((raw/split).exists() for split in splits):
        raise FileExistsError('Preserve each prior physical stage.')
    if args.split=='evaluation':
        gate=read(raw/'development/gate.json')
        if not gate['passed'] or gate['integration_sha256']!=sha(integration_path(p)):
            raise ValueError('The full development gate must pass.')
        if (raw/'evaluation-freeze.json').exists():
            raise FileExistsError('Preserve the original final freeze.')
    space(p,raw,1024**3)
    if args.split=='evaluation':
        write(raw/'evaluation-freeze.json',{'protocol_sha256':sha(args.protocol),'integration_sha256':sha(integration_path(p)),
            'development_gate_sha256':sha(raw/'development/gate.json'),
            'selection':'The unchanged development candidate. All weights, preprocessing, routes, guards, sources and requested goal set stay frozen.'})
    def launch(item):
        split,seed,mode=item
        folder=raw/split/f'{seed}-{mode}'; log=raw/split/f'{seed}-{mode}.log'
        if folder.exists() or log.exists():
            raise FileExistsError('Preserve physical output and log.')
        space(p,log,24*1024**2); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb') as stream:
            process=subprocess.run([sys.executable,'-I',str(Path(__file__)),'--protocol',str(args.protocol.resolve()),
                '--split',split,'--seed',str(seed),'--mode',mode],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=spec['maximum_wall_seconds']+90)
        result=read(folder/'report.json') if (folder/'report.json').exists() else {'status':'harness_error','message':'No terminal report.'}
        row={'split':split,'seed':seed,'mode':mode,'status':result['status'],'message':result['message'],
            'exit_code':process.returncode,'valid':result.get('setup',{}).get('valid',False),
            'destination_m':result.get('destination_m'), 'report_sha256':sha(folder/'report.json') if (folder/'report.json').exists() else None}
        print(row,flush=True)
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=spec['maximum_workers']) as workers:
        for split in splits:
            seeds=p['regression_seeds'] if split=='regression' else p['physical'][split+'_seeds']
            modes=['live'] if split=='regression' else p[split+'_modes']
            for seed in seeds:
                group=list(workers.map(launch,[(split,seed,mode) for mode in modes]))
                rows+=group
                if any(row['status']=='harness_error' or row['exit_code'] for row in group):
                    write(raw/args.split/'stopped.json',{'reason':'Harness error; later seeds remain unexposed.','rows':rows})
                    return 1
    stage=[row for row in rows if row['split']==args.split]
    counts={mode:sum(row['mode']==mode and row['status']=='succeeded' for row in stage) for mode in p[args.split+'_modes']}
    live=[row for row in stage if row['mode']=='live']
    valid=sum(row['valid'] for row in live)
    per_goal=[sum(row['status']=='succeeded' and row['destination_m']==goal for row in live) for goal in p['physical']['destinations_xy_m']]
    limits=p[args.split+'_gate']
    passed=counts['live']>=limits['minimum_live_successes'] and min(per_goal)>=limits['minimum_live_per_destination']
    if args.split=='development':
        passed=passed and counts['live']>counts['baseline'] and len(rows)==16
    else:
        regressions=[row for row in rows if row['split']=='regression']
        passed=passed and counts['live']/max(1,valid)>=limits['minimum_live_fraction_of_valid']
        passed=passed and counts['live']>=counts['baseline']+limits['minimum_more_successes_than_baseline']
        passed=passed and counts['live']>=counts['frozen']+limits['minimum_more_successes_than_frozen']
        passed=passed and sum(row['status']=='succeeded' for row in regressions)==limits['required_exposed_regression_successes'] and len(rows)==102
    write(raw/args.split/'gate.json',{'schema':p['schema'],'protocol_sha256':sha(args.protocol),
        'integration_sha256':sha(integration_path(p)),'passed':bool(passed),'rows':rows,'counts':counts,
        'valid_starts':valid,'live_successes_per_goal':per_goal,'planned_cases_per_mode':len(live),
        'scope':spec['scope'],'standalone_perception_gate_remains_failed':True})
    print({'stage':args.split,'passed':bool(passed),'counts':counts,'valid_starts':valid,'per_goal':per_goal},flush=True)
    return int(not passed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--split',choices=['development','evaluation','regression'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--mode',choices=['live','frozen','baseline'])
    args=parser.parse_args()
    if args.declare:
        declare(args.declare)
    elif args.seed is not None:
        raise SystemExit(trial(args))
    elif args.split in ('development','evaluation'):
        raise SystemExit(batch(args))
    else:
        parser.error('Choose a declaration, a stage, or one declared trial.')

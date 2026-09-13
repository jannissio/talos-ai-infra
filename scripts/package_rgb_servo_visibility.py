"""Recompute saved-state camera diagnostic errors and preserve every outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.rgb_servo_cameras import KEYPOINTS
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    report = json.loads((folder / 'report.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    if sha(folder / 'protocol.json') != report['protocol_sha256'] or sha(folder / 'probe_rgb_servo_visibility.py') != report['source_sha256']:
        raise ValueError('Protocol or diagnostic source changed.')
    for name, digest in {**report['runtime_source_sha256'], **report['input_sha256']}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Reproduction input changed: ' + name)
    model_folder = ROOT / protocol['observer_model']
    if sha(model_folder / 'observer.safetensors') != protocol['observer_sha256']:
        raise ValueError('Observer checkpoint changed.')
    for name, digest in protocol['observer_ir_sha256'].items():
        if sha(model_folder / 'openvino' / name) != digest:
            raise ValueError('Observer IR changed.')
    wanted, truth = set(), {}
    for trace in protocol['traces']:
        source = ROOT / protocol['input_root'] / trace
        model = mujoco.MjModel.from_xml_path(str(source / 'scene.xml'))
        data = mujoco.MjData(model)
        with np.load(source / 'states.npz', allow_pickle=False) as archive:
            states = {name:archive[name] for name in archive.files}
        end = min(protocol['maximum_simulation_time_s'], float(states['time'][-1]))
        indices = np.unique([int(np.argmin(np.abs(states['time']-t))) for t in np.linspace(0,end,protocol['frames_per_trace'])])
        for index in indices:
            data.qpos[:] = states['qpos'][index]
            data.qvel[:] = states['qvel'][index]
            data.time = float(states['time'][index])
            mujoco.mj_forward(model, data)
            points = KEYPOINTS @ data.body('bottle').xmat.reshape(3,3).T + data.body('bottle').xpos
            truth[trace, int(index)] = (points, float(data.time), str(states['stage'][index]))
            wanted.update((trace,int(index),name) for name in protocol['camera_configurations'])
    rows = report['rows']
    if len(rows)!=len(wanted) or {(r['trace'],r['state_index'],r['configuration']) for r in rows}!=wanted:
        raise ValueError('Missing, repeated or undeclared observations.')
    maximum_delta = 0.
    for row in rows:
        points, seconds, stage = truth[row['trace'], row['state_index']]
        if row['simulation_seconds']!=seconds or row['stage']!=stage:
            raise ValueError('Observation is attached to the wrong saved frame.')
        if row['observation']['status']=='observed':
            error = float(np.linalg.norm(np.asarray(row['observation']['keypoints_m'])-points,axis=1).max()*1000)
            maximum_delta=max(maximum_delta,abs(error-row['scoring_only_error_mm']))
        elif row['scoring_only_error_mm'] is not None:
            raise ValueError('A refused observation has an accepted score.')
    if maximum_delta>1e-8:
        raise ValueError('Independent saved-state geometry disagrees with reported errors.')
    missing={(r['trace'],r['state_index']) for r in rows if r['configuration']=='original' and r['observation']['status']!='observed'}
    gate, summaries = protocol['diagnostic_gate'], {}
    for name in protocol['camera_configurations']:
        subset=[r for r in rows if r['configuration']==name]
        accepted=[r for r in subset if r['observation']['status']=='observed']
        errors=[r['scoring_only_error_mm'] for r in accepted]
        recovered=sum((r['trace'],r['state_index']) in missing for r in accepted)
        p95, maximum=(float(np.quantile(errors,.95)),max(errors)) if errors else (None,None)
        summaries[name]={'frames':len(subset),'accepted':len(accepted),'original_refused_frames':len(missing),
                         'recovered_original_refusals':recovered,'accepted_error_p95_mm':p95,'accepted_error_max_mm':maximum,
                         'diagnostic_gate_passed':bool(missing and errors and len(accepted)/len(subset)>=gate['minimum_overall_acceptance_fraction']
                           and recovered/len(missing)>=gate['minimum_recovery_fraction_on_original_refused_frames']
                           and p95<=gate['maximum_accepted_p95_error_mm'] and maximum<=gate['maximum_accepted_error_mm'])}
    eligible=[name for name in summaries if name!='original' and summaries[name]['diagnostic_gate_passed']]
    selected=min(eligible,key=lambda name:(-summaries[name]['recovered_original_refusals'],summaries[name]['accepted_error_p95_mm'])) if eligible else None
    if summaries!=report['summaries'] or selected!=report['selected_configuration']:
        raise ValueError('Independent aggregation differs from the recorded selection.')
    return {'all_observations_verified':len(rows),'saved_states_verified':len(truth),
            'maximum_scoring_discrepancy_mm':maximum_delta,'summaries':summaries,'selected_configuration':selected,
            'physical_trials':0,'promoted':False,'scope':'Exposed saved-state camera diagnostic; fresh perception and physical validation remain required.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        manifest=json.loads((args.output/'manifest.json').read_text())
        for row in manifest['files']:
            if sha(args.output/row['file'])!=row['sha256']:
                raise ValueError('Packaged file changed: '+row['file'])
        result=audit(args.output)
        if result!=json.loads((args.output/'audit.json').read_text()):
            raise ValueError('Saved audit changed.')
    else:
        if not args.source:
            parser.error('--source is required for a new package')
        if args.output.exists():
            raise FileExistsError('Preserve the previous evidence package.')
        result=audit(args.source)
        files=[args.source/name for name in ('report.json','protocol.json','probe_rgb_servo_visibility.py')]+sorted(args.source.glob('*.png'))
        total=sum(p.stat().st_size for p in args.source.rglob('*') if p.is_file())+sum(p.stat().st_size for p in files)+2*1024**2
        protocol=json.loads((args.source/'protocol.json').read_text())
        if total>protocol['budget']['maximum_new_data_mib']*1024**2:
            raise ValueError('Raw plus packaged evidence exceeds the diagnostic data budget.')
        require_space(args.output,total)
        args.output.mkdir(parents=True)
        entries=[]
        for source in files:
            target=args.output/source.name
            require_space(target,source.stat().st_size+1024**2)
            target.write_bytes(source.read_bytes())
            entries.append({'file':target.name,'sha256':sha(target),'bytes':target.stat().st_size,'transformation':'byte-identical copy'})
        if audit(args.output)!=result:
            raise ValueError('Packaged diagnostic changed.')
        for name,value in [('audit.json',result),('manifest.json',{'files':entries,'source':args.source.as_posix(),'originals_modified':False})]:
            require_space(args.output/name,1024**2)
            (args.output/name).write_bytes((json.dumps(value,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result))

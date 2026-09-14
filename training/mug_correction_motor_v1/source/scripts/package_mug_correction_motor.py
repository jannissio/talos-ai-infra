"""Preserve the complete stopped local-motor data gate, without inventing a fit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.mug_correction_motor_v1 import read,sha,space,write,data_gate


def package():
    protocol=ROOT/'docs/robotics/experiments/mug-correction-motor-v1.json'
    p=read(protocol)
    raw=ROOT/p['raw_root']
    training,evidence=ROOT/p['training_package'],ROOT/p['evidence_package']
    if any(path.exists() for path in (training,evidence,ROOT/p['model_package'])):
        raise FileExistsError('Preserve existing packages.')
    if any((raw/name).exists() for name in ('fit','openvino','evaluation')):
        raise ValueError('A stopped input gate must not consume later fit/export/fresh stages.')
    sources,payload,summary={},[],{}
    for split in ('training','development'):
        manifest,arrays=data_gate(p,raw/split,protocol)
        audit=read(raw/split/'audit.json')
        if manifest['label_coverage_passed'] or not audit['passed'] or audit['data_sha256']!=sha(raw/split/'data.npz'):
            raise ValueError('Expected an independently verified, failed coverage gate for each initial split.')
        if len(arrays['joints'])!=p['data'][split+'_states'] or len(manifest['rows'])!=len(arrays['joints']):
            raise ValueError('Every attempted input must be retained.')
        bad=[r for r in manifest['rows'] if not r['accepted']]
        summary[split]={'attempted':len(arrays['joints']),'accepted':int(arrays['accepted'].sum()),'rejected':len(bad),
            'coverage_fraction':float(arrays['accepted'].mean()),'gate_passed':False,
            'rejected_excessive_joint_delta':sum(max(abs(np.asarray(r['joint_delta_rad'])))>p['data']['maximum_joint_step_rad'] for r in bad),
            'rejected_position_error':sum(r['position_error_mm']>p['data']['maximum_label_position_error_mm'] for r in bad),
            'rejected_axis_error':sum(r['axis_error']>p['data']['maximum_label_axis_error'] for r in bad),
            'maximum_joint_delta_rad':max(max(abs(np.asarray(r['joint_delta_rad']))) for r in manifest['rows']),
            'p95_position_error_mm':float(np.quantile([r['position_error_mm'] for r in manifest['rows']],.95)),
            'maximum_position_error_mm':max(r['position_error_mm'] for r in manifest['rows'])}
        # Convert NumPy scalar counts before JSON serialization.
        for key in ('rejected_excessive_joint_delta','rejected_position_error','rejected_axis_error'):
            summary[split][key]=int(summary[split][key])
        sources.update(manifest['source_sha256'])
        payload.extend(((raw/split/'data.npz',training/split/'data.npz'),
                        (raw/split/'manifest.json',evidence/split/'manifest.json'),
                        (raw/split/'audit.json',evidence/split/'audit.json')))
        for kind in ('data','audit'):
            name=f'mug-correction-motor-v1-{split}-{kind}.log'
            payload.append((ROOT/'.run/final-goal'/name,evidence/'console'/name))
    sources[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
        if name.endswith('.py'):payload.append((ROOT/name,training/'source'/name))
    for root in (training,evidence):payload.append((protocol,root/'protocol.json'))
    preflight=space(p,evidence,sum(source.stat().st_size for source,_ in payload)+4*1024**2)
    for source,target in payload:
        space(p,target,source.stat().st_size+1024)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
    write(p,training/'reproduction-inputs.json',{'source_sha256':sources,
        'data_sha256':{split:sha(training/split/'data.npz') for split in summary},
        'attempts':'Every sampled episode/phase/joint perturbation, matrix, translation, endpoint and label acceptance is retained. Full per-attempt FK results and independent audits are in the evidence package.',
        'model_fitted':False,'fresh_seed_unexposed':p['data']['evaluation_seed']})
    write(p,evidence/'data-gate.json',{'schema':p['schema'],'protocol_sha256':sha(protocol),'source_sha256':sha(Path(__file__)),
        'splits':summary,'required_coverage':p['data']['minimum_label_coverage'],'preflight':preflight,
        'status':'stopped_at_input_gate','independent_geometry_audits_passed':True,'model_fitted':False,
        'export_performed':False,'new_physical_trials':0,'selected_robot_unchanged':True,
        'fresh_seed_unexposed':p['data']['evaluation_seed'],
        'diagnosis':'Every rejected case exceeds the 0.08 rad joint-step limit; a subset also exceeds the 0.15 mm endpoint-error limit. Fixed 2 mm Cartesian steps are unsuitable for part of this sampled placement region. The audit verifies these labels; it does not make them safe motor commands.',
        'next_hypothesis':'Separately declare joint-limited correction sizing before fitting; do not relax this failed protocol or drop inconvenient cases.'})
    counts={}
    for key,root in (('training',training),('evidence',evidence)):
        files={f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()}
        size=sum((root/name).stat().st_size for name in files)
        write(p,root/'manifest.json',{'files':files,'bytes':size,'schema':p['schema']})
        counts[key]={'files':len(files),'bytes':size}
    print({'status':'stopped_at_input_gate','splits':summary,'packages':counts,'model_fitted':False},flush=True)


if __name__=='__main__':package()

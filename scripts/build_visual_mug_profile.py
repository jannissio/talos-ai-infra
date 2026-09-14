"""Create a compact runtime profile only from fully passing physical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    protocol=read(args.protocol);evidence=args.evidence or ROOT/protocol['evidence_package']
    output=args.output
    if output.exists():raise FileExistsError('Preserve the earlier promoted profile.')
    manifest=read(evidence/'manifest.json')
    required=('protocol.json','development-audit.json','evaluation-audit.json','development-freeze.json','evaluation-freeze.json')
    for name in required:
        if sha(evidence/name)!=manifest['files'][name]:raise ValueError('Promotion evidence differs from its package manifest.')
    if sha(evidence/'protocol.json')!=sha(args.protocol):raise ValueError('Packaged protocol differs.')
    reports={split:read(evidence/(split+'-audit.json')) for split in ('development','evaluation')}
    for split,report in reports.items():
        planned=len(protocol[split+'_seeds'])*len(protocol['starting_presets'])*len(protocol['modes'])
        if report['completed']!=planned or not report['gate_passed'] or not all(report['requirements'].values()):
            raise ValueError('Both complete physical gates must pass before promotion.')
    frozen=read(evidence/'development-freeze.json')
    final=read(evidence/'evaluation-freeze.json')
    if final['inputs']!=frozen['inputs'] or final['development_audit_sha256']!=sha(evidence/'development-audit.json'):
        raise ValueError('Final selection does not match passing development.')
    # UI/server/hosting entry points change to expose this option. The evaluated
    # motor, cameras, guard, tasks, physics and monitor retain their exact bytes.
    deployment_entries={'simulation_lab/engine.py','simulation_lab/public_trial.py','simulation_lab/server.py'}
    sources={name:digest for name,digest in frozen['inputs']['source_sha256'].items()
        if name.startswith('simulation_lab/') and name not in deployment_entries}
    for name,digest in {**sources,**frozen['inputs']['model_sha256']}.items():
        if sha(ROOT/name)!=digest:raise ValueError('An evaluated runtime/model dependency changed: '+name)
    payloads={name:(evidence/name).read_bytes() for name in required}
    preflight=require_space(output,sum(map(len,payloads.values()))+1024**2);output.mkdir(parents=True)
    for name,payload in payloads.items():
        require_space(output/name,len(payload)+1024)
        with (output/name).open('xb') as stream:stream.write(payload)
    profile={'schema':'talos.promoted-mug-vision.v1','name':'dinner_visual_mug_v1',
        'protocol_sha256':sha(args.protocol),'evidence_package':protocol['evidence_package'],
        'evidence_manifest_sha256':sha(evidence/'manifest.json'),
        'evidence_files':{name:sha(output/name) for name in required},
        'runtime_sources':sources,'model_files':frozen['inputs']['model_sha256'],
        'final_successes':{key:reports['evaluation'][key] for key in ('passed_workflows','passed_mugs','gate_passed')},
        'preflight':preflight,'builder_sha256':sha(__file__),
        'scope':'Live stereo correction during late mug placement only. Other skills retain initial image conditioning and motor feedback. Finite evaluated starting presets; no arbitrary placement, pouring or airborne handoff claim.'}
    payload=(json.dumps(profile,indent=2)+'\n').encode();require_space(output/'profile.json',len(payload)+1024)
    with (output/'profile.json').open('xb') as stream:stream.write(payload)
    print({'profile':output.name,'files':len(payloads)+1,'final_successes':profile['final_successes']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v2.json')
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'models/dinner_visual_mug_v1')
    args=parser.parse_args();build(args)

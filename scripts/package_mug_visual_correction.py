"""Preserve complete mug comparisons, failed attempts and portable state traces."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mug_visual_correction import read,sha,space
from scripts.audit_mug_visual_correction import audit


def encoded(value):return (json.dumps(value,indent=2)+'\n').encode()


def redact(payload):
    for path,label in ((ROOT,'<repository>'),(Path(sys.prefix),'<runtime>'),(Path(sys.base_prefix),'<python>')):
        for name in (str(path),path.as_posix(),str(path).replace('\\','\\\\')):
            payload=payload.replace(name.encode(),label.encode())
    return payload


def compare_derived(original,derived,root):
    for key,value in original.items():
        if key in ('audit_source_sha256','rows'):continue
        if derived[key]!=value:raise ValueError('Recounted evidence disagrees: '+key)
    for before,after in zip(original['rows'],derived['rows']):
        for key,value in before.items():
            if key=='scene_sha256':
                source=root/Path(before['report']).parent/'original-scene.xml'
                if source.exists() and sha(source)!=value:raise ValueError('Original scene bytes changed.')
                if not source.exists() and after[key]!=value:raise ValueError('An original raw scene changed.')
            elif after[key]!=value:raise ValueError('A recounted trial disagrees: '+key)


def verify(p,protocol,evidence):
    manifest=read(evidence/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(evidence/name)!=digest:raise ValueError('Package artifact changed: '+name)
    if sha(evidence/'protocol.json')!=sha(protocol):raise ValueError('The packaged protocol differs.')
    summaries={}
    for split in manifest['splits']:
        original=read(evidence/(split+'-audit.json'))
        derived=audit(p,protocol,evidence,split)
        compare_derived(original,derived,evidence)
        summaries[split]={k:derived[k] for k in ('completed','trace_frames','portable_scenes_loaded','passed_workflows','passed_mugs','gate_passed')}
    return {'passed':True,'manifest_files_checked':len(manifest['files']),'manifest_sha256':sha(evidence/'manifest.json'),
        'splits':summaries,'validation_source_sha256':{name:sha(ROOT/name) for name in
            ('scripts/package_mug_visual_correction.py','scripts/audit_mug_visual_correction.py','scripts/package_spoon_release.py')},
        'scope':'Package-only recount, every original report/state retained, all portable scenes loaded, original scene bytes and frozen models/assets verified. No new physical trial.'}


def package(p,protocol,evidence):
    raw=ROOT/p['raw_root'];name=raw.name
    if evidence.exists():raise FileExistsError('Preserve earlier evidence packages.')
    development=read(raw/'development-audit.json')
    splits=['development']
    if development['gate_passed']:
        final=read(raw/'evaluation-audit.json');splits.append('evaluation')
    elif (raw/'evaluation').exists() or (raw/'evaluation-freeze.json').exists():
        raise ValueError('Failed development must preserve unexposed final scenes.')
    for split in splits:
        derived=audit(p,protocol,raw,split)
        compare_derived(read(raw/(split+'-audit.json')),derived,raw)
    inputs=[]
    for source in sorted(raw.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
            inputs.append((source,source.relative_to(raw)))
    inputs.append((protocol,Path('protocol.json')))
    for source in sorted((ROOT/'.run/final-goal').glob(name+'*.log')):
        if not re.search(r'-(package|verify)(?:[.-]|$)',source.name):
            inputs.append((source,Path('console')/source.name))
    # V1's failed setup has an independent saved-state, no-step diagnosis.
    diagnosis=ROOT/'.run/mug-renderer-lifecycle-v1'
    if name=='mug-visual-correction-v1':
        for source in sorted(diagnosis.iterdir()):
            if source.is_file():inputs.append((source,Path('renderer-diagnostic')/source.name))
        inputs.append((ROOT/'.run/final-goal/mug-renderer-lifecycle-v1.log',Path('console/mug-renderer-lifecycle-v1.log')))
    helpers=('scripts/package_mug_visual_correction.py','scripts/audit_mug_visual_correction.py',
        'scripts/package_spoon_release.py','scripts/diagnose_mug_renderer.py')
    for source in helpers:inputs.append((ROOT/source,Path('validation-source')/source))
    estimate=sum(source.stat().st_size for source,_ in inputs)+8*1024**2
    preflight=space(p,evidence,estimate);evidence.mkdir(parents=True)
    provenance=[]
    for source,relative in inputs:
        target=evidence/relative;payload=source.read_bytes();transformation='byte-identical copy'
        if source.name=='scene.xml':
            space(p,target,len(payload)*2+1024)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.with_name('original-scene.xml').open('xb') as stream:stream.write(payload)
            tree=ET.fromstring(payload);compiler=tree.find('compiler')
            meshdir=(source.parent/compiler.get('meshdir')).resolve()
            if not meshdir.is_relative_to(ROOT/'simulation_lab/assets'):raise ValueError('Unexpected scene asset path.')
            compiler.set('meshdir',os.path.relpath(meshdir,target.parent).replace('\\','/'))
            payload=ET.tostring(tree,encoding='utf-8');transformation='Portable mesh directory only; original XML also retained.'
        elif source.suffix in ('.json','.log','.py','.md','.txt'):
            cleaned=redact(payload)
            if cleaned!=payload:payload=cleaned;transformation='Private filesystem prefixes replaced; original local digest retained.'
        space(p,target,len(payload)+1024);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(payload)
        provenance.append({'file':relative.as_posix(),'source':source.relative_to(ROOT).as_posix(),
            'source_sha256':sha(source),'packaged_sha256':sha(target),'transformation':transformation})
    selected=read(raw/('evaluation-audit.json' if len(splits)>1 else 'development-audit.json'))
    failed=[key for key,value in selected['requirements'].items() if not value]
    table=['| Mode | Standard workflows | Left-reach workflows | Standard mugs | Left-reach mugs |',
        '| --- | --- | --- | --- | --- |']
    denominator=len(p[selected['split']+'_seeds'])
    for mode in p['modes']:
        c=selected['passed_workflows'][mode];m=selected['passed_mugs'][mode]
        table.append(f'| {mode} | {c["upright"]}/{denominator} | {c["wide_left"]}/{denominator} | {m["upright"]}/{denominator} | {m["wide_left"]}/{denominator} |')
    note=('All 24 experimental variants fail before the first movement because renderer replacement produces black camera frames. '
        'The unchanged baseline passes all 12 complete workflows. No mug correction was attempted. '
        'The independent renderer diagnostic uses an exposed saved state, with physics stepping disabled: closing the old renderer before '
        'creating the replacement exactly restores all three RGB images and the initial neural output. '
        'That setup repair requires a separate protocol; it does not repair or promote this stopped result.'
        if name.endswith('-v1') else 'This separately declared comparison changes renderer setup order only. The V1 mug controller, observer, motor, timing and physical criteria remain unchanged. All failures and denominators are retained.')
    readme=f'''# {name}: complete physical comparison

{note}

{chr(10).join(table)}

The latest completed split is **{selected['split']}**: {selected['completed']} attempted workflows, {selected['trace_frames']:,} retained state frames, {selected['correction_queries']} visual correction queries. Its gate **{'passes' if selected['gate_passed'] else 'fails'}**. Failed checks: {', '.join(failed) or 'none'}.

The [protocol](protocol.json), [development results](development-audit.json), every scene/report/state trace, exact frozen runtime sources and console provenance are retained. Original XML files are stored beside the portable scenes. Package verification recounts outcomes and compares prefixes before the first correction, preserving mismatches as a failed criterion. Earlier missing-test-runner evidence is kept where applicable; no test ran during that failed launcher.

Models remain in [the stereo observer package](../../../../models/mug_keypoint_observer_v1/README.md) and [the limited motor package](../../../../models/mug_correction_motor_v2/README.md). Their manifests and the selected dinner suite are frozen in development-freeze.json. This experiment creates no replacement weights. The selected dinner/browser/hosted baseline remains unchanged pending a separate promotion decision.

Run the package-only integrity and paired-result check from the repository:

```powershell
.\\.venv-training\\Scripts\\python scripts/package_mug_visual_correction.py --protocol docs/robotics/experiments/{name}.json --verify-only
```

No broad workspace, arbitrary-language, Intel-hardware or human-voice claim follows from this finite comparison. GitHub and Hugging Face remain private until the final release. Final Submit is exclusively the user's action.
'''
    extra={'copy-provenance.json':encoded({'preflight':preflight,'entries':provenance}),
           'README.md':readme.encode()}
    for name,payload in extra.items():
        target=evidence/name;space(p,target,len(payload)+1024)
        with target.open('xb') as stream:stream.write(payload)
    files={path.relative_to(evidence).as_posix():sha(path) for path in sorted(evidence.rglob('*')) if path.is_file()}
    payload=encoded({'schema':p['schema'],'splits':splits,'files':files,'bytes':sum(path.stat().st_size for path in evidence.rglob('*') if path.is_file()),
        'scope':'All declared attempts and original frozen source/model identities retained. This manifest excludes later completion records.'})
    space(p,evidence/'manifest.json',len(payload)+1024)
    with (evidence/'manifest.json').open('xb') as stream:stream.write(payload)
    result=verify(p,protocol,evidence)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();args.protocol=args.protocol.resolve();p=read(args.protocol);evidence=ROOT/p['evidence_package']
    if args.verify_only:print(verify(p,args.protocol,evidence),flush=True)
    else:package(p,args.protocol,evidence)

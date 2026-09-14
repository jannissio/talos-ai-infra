"""Check the published spoon package in an isolated minimal source checkout."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from scripts.package_spoon_release import budget,read
from simulation_lab.storage import require_space
BASE=ROOT/'.run/spoon-release-v1'
target=BASE/'portable-checkout'
log=BASE/'portability-restore.log'
trial_log=BASE/'portability-trial.log'
result_path=BASE/'portability.json'
if any(p.exists() for p in (target,log,trial_log,result_path)):
    raise FileExistsError('All portable-checkout, result and log paths must be unused.')
evidence=ROOT/'docs/robotics/evidence/spoon-release-v1'
frozen=read(evidence/'development/frozen-inputs.json')
manifest=read(evidence/'package-manifest.json')
files={r['file'] for r in manifest['files']}
files.update(frozen['inputs']['source_sha256'])
files.update(frozen['inputs']['asset_sha256'])
for selection in frozen['inputs']['suites'].values():
    files.update(p for p in selection['files_sha256'] if not p.startswith('.run/'))
files.update(['docs/robotics/experiments/spoon-release-v1.json','scripts/package_spoon_release.py',
              'docs/robotics/evidence/spoon-release-v1/package-manifest.json',
              'training/spoon_release_v1/input-replay.json'])
expected=sum((ROOT/name).stat().st_size for name in files)+16*1024**2
space=budget(expected)
target.mkdir(parents=True)
for name in sorted(files):
    payload=(ROOT/name).read_bytes();out=target/name
    require_space(out,len(payload)+1024**2)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:stream.write(payload)
with log.open('x',encoding='utf-8') as stream:
    restore=subprocess.run([sys.executable,str(target/'scripts/package_spoon_release.py'),'--restore-evaluation-inputs'],cwd=target,stdout=stream,stderr=subprocess.STDOUT)
if restore.returncode:
    print('Portable restore failed; logs and all copies retained.',flush=True)
    raise SystemExit(restore.returncode)
out=target/'.run/spoon-release-v1/portable-trial-9202'
require_space(out,12*1024**2)
with trial_log.open('x',encoding='utf-8') as stream:
    trial=subprocess.run([sys.executable,str(target/'scripts/evaluate_spoon_release.py'),'trial','--freeze',
      'docs/robotics/evidence/spoon-release-v1/development/frozen-inputs.json','--seed','2026099202',
      '--preset','upright','--controller','candidate','--output',str(out)],cwd=target,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
original=read(evidence/'development/2026099202-upright-candidate/report.json')
replayed=read(out/'report.json')
original_metrics={r['skill']:r['metrics'] for r in original['task']['results']}
replayed_metrics={r['skill']:r['metrics'] for r in replayed['task']['results']}
result={'scope':'Exposed development reproduction from an isolated minimal checkout, not a new selection trial.',
        'files_copied':len(files),'copy_bytes':sum((ROOT/name).stat().st_size for name in files),
        'disk_preflight':space,'restore_exit':restore.returncode,'trial_exit':trial.returncode,
        'seed':2026099202,'preset':'upright','controller':'candidate','status':replayed['status'],
        'all_six_physical_metrics_identical':original_metrics==replayed_metrics,
        'initial_state_identical':original['initial_state_sha256']==replayed['initial_state_sha256'],
        'simulation_seconds':replayed['simulation_seconds'],'wall_seconds':replayed['wall_seconds'],
        'original_report_sha256':hashlib.sha256((evidence/'development/2026099202-upright-candidate/report.json').read_bytes()).hexdigest()}
result['passed']=trial.returncode==0 and result['all_six_physical_metrics_identical'] and result['initial_state_identical']
require_space(result_path,1024**2)
with result_path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)

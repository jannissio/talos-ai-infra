"""Rebuild compact evidence without relabeling familiar starts as held-out tests."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['episode','checkpoint','success','reason','max_lift_cm','best_supported_hold_s','stable_release_s',
        'placement_error_m','simulated_s','other_object_max_displacement_m','parked_arm_max_motion_deg',
        'longest_unsupported_contact_gap_s','fault_injection','policy_details','teacher_updates','state_resets']

def main(complete=False):
    rows=[];suites=[]
    for path in sorted((ROOT/'.run').glob('bottle-sequence*.json')):
        data=json.loads(path.read_text())
        if 'success' in data and 'checkpoint' in data:
            rows.append({'result_file':path.relative_to(ROOT).as_posix(),**{k:data[k] for k in FIELDS if k in data}})
    for path in sorted((ROOT/'.run').glob('bottle-sequence*/summary.json')):
        data=json.loads(path.read_text())
        suites.append({'result_file':path.relative_to(ROOT).as_posix(),'suite':data['suite'],
                      'successes':data['successes'],'trials':data['trials'],
                      'results':[{k:r[k] for k in FIELDS if k in r} for r in data['results']]})
    used=sum(f.stat().st_size for p in (ROOT/'.run').glob('bottle-sequence*') for f in (p.rglob('*') if p.is_dir() else [p]) if f.is_file())
    report={'goal_status':'complete' if complete else 'active','individual_development_trials':rows,'suites':suites,
            'sequence_artifacts_bytes':used,'budget_bytes':2*1024**3,'free_bytes':shutil.disk_usage(ROOT).free,
            'reserve_bytes':10*1024**3,'images_copied_for_training':0,
            'interpretation':'Physical evaluator unchanged. Neural familiar success and programmed tracking recovery do not prove arbitrary-position generalization.'}
    (ROOT/'docs/robotics/bottle-sequence-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'individual_trials':len(rows),'suites':len(suites),'artifact_mib':round(used/1024**2,1),'free_gib':round(report['free_bytes']/1024**3,1)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');main(p.parse_args().complete)

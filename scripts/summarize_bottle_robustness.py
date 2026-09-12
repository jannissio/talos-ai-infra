"""Record the frozen result, including regressions and a failed promotion decision."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def compact(row):
    keys=['episode','success','reason','max_lift_cm','best_supported_hold_s','placement_error_m',
          'simulated_s','other_object_max_displacement_m','parked_arm_max_motion_deg',
          'longest_unsupported_contact_gap_s','teacher_updates','state_resets']
    return {k:row[k] for k in keys if k in row}
def main():
    reports={name:read('.run/bottle-robustness-'+folder+'/summary.json') for name,folder in [
        ('baseline_development','baseline-dev'),('candidate_development','candidate-dev'),
        ('candidate_training','candidate-training'),('candidate_familiar','selected-familiar'),
        ('baseline_held_out','baseline-test'),('candidate_held_out','selected-test')]}
    baseline=reports['baseline_held_out'];candidate=reports['candidate_held_out']
    promotion=(reports['candidate_development']['successes']>reports['baseline_development']['successes']
               and reports['candidate_familiar']['successes']==5 and candidate['successes']>baseline['successes'])
    lineage=read('.run/bottle-robustness-data/fit-lineage.json')
    protocol=read('docs/robotics/bottle-robustness-protocol.json')
    assert not {r['source'] for r in lineage['episodes']} & {r['id'] for r in protocol['held_out']}
    episodes=[]
    for spec in protocol['training']:
        folder=ROOT/'.run/bottle-robustness-data'/spec['id'];manifest=json.loads((folder/'manifest.json').read_text())
        episodes.append({'episode':spec['id'],'eligible':manifest['training_eligible'],
            'frames':manifest['observation_count']-1,'images':manifest['images']['count'],
            'replays':manifest['replays'],'gripper_cap_nm':manifest['gripper_torque_cap_nm']})
    artifacts=[]
    for path in (ROOT/'.run').glob('bottle-robustness*'):
        size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size
        artifacts.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'bytes':size})
    total=sum(r['bytes'] for r in artifacts)
    assert total<protocol['budget_bytes']
    report={'promotion':promotion,'browser_checkpoint_retained':'.run/retrieval-fit5-v4' if not promotion else None,
        'scope':'Small local upright neighborhood; four untouched test positions, not full-workspace generalization.',
        'results':{name:{'successes':r['successes'],'trials':r['trials'],'episodes':[compact(x) for x in r['results']]} for name,r in reports.items()},
        'new_demonstrations':episodes,'training_observations_total':sum(r['frames'] for r in lineage['episodes']),
        'held_out_excluded_from_training':True,'freeze_sha256':hashlib.sha256((ROOT/'docs/robotics/bottle-robustness-freeze.json').read_bytes()).hexdigest(),
        'actuator_pause':compact(read('.run/bottle-robustness-selected-stall.json')),
        'artifact_bytes':total,'artifact_budget_bytes':protocol['budget_bytes'],
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'artifacts':artifacts,
        'next_direction':'Keep the successful baseline; use sequence-aware learning and corrective feedback rather than further nearest-trajectory parameter tuning. These four test poses must not be reused as an untouched test set after inspecting their outcomes.'}
    (ROOT/'docs/robotics/bottle-robustness-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'promoted':promotion,'artifact_mib':total/1024**2,'free_gib':report['disk_free_bytes']/1024**3}))
if __name__=='__main__':main()

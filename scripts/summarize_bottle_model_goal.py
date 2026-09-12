"""Compact, auditable outcome summary; never promotes familiar trials to held-out results."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['episode','success','reason','simulated_s','max_lift_cm','best_supported_hold_s',
        'stable_release_s','placement_error_m','other_object_max_displacement_m',
        'parked_arm_max_motion_deg','longest_unsupported_contact_gap_s',
        'teacher_updates','state_resets','rejection_detail','fault_injection']
def read(path):return json.loads((ROOT/path).read_text())
def compact(row):return {key:row[key] for key in FIELDS if key in row}
def main():
    familiar=read('.run/retrieval-v4-familiar/summary.json')
    probes=read('.run/retrieval-v4-probes/summary.json')
    checkpoint=ROOT/'.run/retrieval-fit5-v4'
    score=read('.run/act-relative-upright-score.json')
    report={'scope':'Demonstration-fitted retrieval baseline; not neural ACT or a VLA. Five familiar starts; development probes are not held-out evaluation.',
        'selected_checkpoint':str(checkpoint.relative_to(ROOT)),
        'familiar':{'successes':familiar['successes'],'trials':familiar['trials'],'results':[compact(r) for r in familiar['results']]},
        'development_probes':{'successes':probes['successes'],'trials':probes['trials'],'results':[compact(r) for r in probes['results']]},
        'mid_approach_actuator_pause':compact(read('.run/retrieval-v4-midstall.json')),
        'blank_cameras':compact(read('.run/retrieval-v4-blank.json')),
        'relative_ACT':{'physical':compact(read('.run/act-relative-upright-rollout.json')),
            'offline_scores':{k:score[k] for k in ['first_five_left_arm_mae_rad','left_arm_mae_rad','left_gripper_mae_rad','first_five_current_joint_reference_mae_rad']},
            'finite_updates':1000,'training_elapsed_seconds':1317.10},
        'checkpoint_bytes':sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
        'checkpoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'simulation_lab/retrieval_policy.py',ROOT/'scripts/evaluate_act.py',ROOT/'simulation_lab/policy_control.py']},
        'disk_free_gib':shutil.disk_usage(ROOT).free/1024**3,
        'artifacts_note':'Full per-cycle traces and videos remain under .run; no duplicated training images. Earlier failed experiments preserved.'}
    (ROOT/'docs/robotics/bottle-model-goal-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['checkpoint_bytes','disk_free_gib']}))
if __name__=='__main__':main()

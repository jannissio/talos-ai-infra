"""Make explicit the limited task implications of initial-pose reach certificates."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put
prior=ROOT/'docs/robotics/evidence/analytic-contact-reach-v1.json'
out=ROOT/'docs/robotics/evidence/analytic-contact-reach-v2.json';log=ROOT/'.run/analytic-contact-reach-v2-export.log'
if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(out,8*1024**2)
with log.open('x') as stream:
    report=json.loads(prior.read_text());report['schema']='analytic-contact-reach-v2';report['preflight']=preflight
    report['scope_qualifications']=[
        'Initial direct-contact separation means no collision-enabled arm surface can touch the object in its saved initial pose. It is not a global task-impossibility theorem under object-assisted indirect rearrangement.',
        'Initial simultaneous-pinch separation means at least one required fixed/moving finger group is excluded at that initial pose. A moving-finger nudge or push of the manipulated item may bring it into later pinch range. An enclosing-ball overlap for the moving finger is not itself a reachable-contact certificate.',
        'The4mm envelopes are explicitly hypothetical bounds on object-body translation. The existing4mm unrelated-item guard does not limit deliberate translation of the item currently being manipulated. Therefore surviving that envelope does not exclude a larger permitted contact-driven pregrasp repositioning.',
        'The arbitrary-rotation certificate for4002 plate also remains conditional on its body origin staying within4mm of its saved position. It does not cover arbitrary cumulative displacement or indirect use of another object.',
        'No sampler eligibility, start state, scene denominator, held-out set, task tolerance or physics changed. All original scenes remain retained. The4004 side_plate initial pinch gap can motivate a separately guarded pregrasp repositioning experiment, not rejection of the scene.'
    ]
    report['classification_summary']=dict(initial_direct_contact_separation=[dict(seed=2026114001,object='glass'),dict(seed=2026114001,object='spoon'),dict(seed=2026114002,object='plate')],additional_initial_simultaneous_pinch_separation=[dict(seed=2026114002,object='bottle'),dict(seed=2026114004,object='side_plate')],arbitrary_rotation_and_4mm_translation_no_arm_contact=[dict(seed=2026114002,object='plate')],global_task_impossibility_claimed=False,sampler_eligibility_changed=False,original_scene_denominators_changed=False)
    report['hashes_sha256'][prior.relative_to(ROOT).as_posix()]=hashlib.sha256(prior.read_bytes()).hexdigest()
    report['hashes_sha256'][Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    put(out,report);stream.write('Export complete.\n')

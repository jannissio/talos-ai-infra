"""Final bridge package adds generic IK branch checks to preserved physical V1."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put
out=ROOT/'docs/robotics/evidence/glass-bridge-development-v2.json';log=ROOT/'.run/glass-bridge-development-v2-export.log'
if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(out,8*1024**2)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    report=json.loads((ROOT/'docs/robotics/evidence/glass-bridge-development-v1.json').read_text());report['schema']='glass-bridge-development-v2';report['storage_preflight']=preflight
    check=ROOT/'.run/glass-bridge-generic-seed-v1/report.json';report['generic_seed_check']=json.loads(check.read_text())
    report['integration_recommendations'][3]='horizontal_glass_diameter_candidates(data,name,candidate_type) returns the physically used67mm band first and unverified72mm band, exact proof001 tool point/local body-up axis, opening.85 and torque.65Nm, no seed/position filters. horizontal_glass_ik_seeds() returns generic[0,.8,.4,-1.2,roll] seeds for rolls−2.4,0,+2.4; the first two pass actual post-bridge source/approach and measured-held goal/route checks, +2.4 is preserved as unsolved. These generic checks used no scene-specific initial joint vector and no additional physical trial. The successful physical run used the earlier proof-derived initial guess; a continuous shared-workflow rerun remains required after integration.'
    for path in (check,ROOT/'scripts/verify_glass_bridge_generic_seed.py',ROOT/'simulation_lab/horizontal_glass_grasp_candidates.py',Path(__file__)):
        report['hashes_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert ':\\' not in json.dumps(report) and 'C:/' not in json.dumps(report)
    put(out,report);print('Export complete.')

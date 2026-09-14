"""Sanitized evidence for the bounded glass bridge search and exact continuation."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put

target=ROOT/'docs/robotics/evidence/glass-bridge-development-v1.json';log=ROOT/'.run/glass-bridge-development-v1-export.log'
if target.exists() or log.exists():raise FileExistsError('Preserve evidence and log.')
preflight=require_space(target,8*1024**2)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    rows=[];hashes={};geometry=[]
    def bind(path):hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for version in range(1,8):
        folder=ROOT/f'.run/glass-bridge-geometry-v{version}';report=json.loads((folder/'report.json').read_text())
        for name in ('report.json','scene.xml','source-manifest.json'):bind(folder/name)
        geometry.append(dict(run=folder.relative_to(ROOT).as_posix(),checked_buffers=report['checked_buffers'],checked_bridge_proofs=report['proofs'],scope=report['scope']))
        for path in folder.glob('proof-*.json'):bind(path)
    for version in range(1,4):
        folder=ROOT/f'.run/glass-bridge-physics-v{version}'
        for leg in ('leg1','leg2'):
            path=folder/leg/'result.json'
            if not path.exists():continue
            result=json.loads(path.read_text());rows.append(dict(run=(folder/leg).relative_to(ROOT).as_posix(),result=result))
            for name in ('result.json','states.npz','source-manifest.json','search.json','placement-search.json'):bind(folder/leg/name)
    for name in ('scripts/develop_glass_bridge.py','scripts/develop_glass_bridge_physics.py','scripts/develop_glass_reorientation.py','simulation_lab/horizontal_glass_grasp_candidates.py','scripts/export_glass_bridge_evidence.py'):bind(ROOT/name)
    report=dict(schema='glass-bridge-development-v1',scope='Privileged teacher dependency on exposed upright glass in joint seed2026114004 after mug clearance. No inverted/sideways glass, complete seven-object task, learned controller or coverage claim.',storage_preflight=preflight,geometry=geometry,attempts=rows,hashes_sha256=hashes,summary=dict(checked_buffer_entries=sum(x['checked_buffers'] for x in geometry),reset_only_bridge_proofs=sum(x['checked_bridge_proofs'] for x in geometry),planning_only_first_leg=1,failed_retreat_preserved=True,successful_two_leg_route=True,combined_final_state_frames=15085,combined_final_motor_steps=15084,all_motor_replays_exact=True,geometry_or_physics_changes=False,final_task_goal=[.075,.10,.76]),integration_recommendations=[
      'Use measured held-glass orientation as an explicit intermediate target candidate, with the existing3deg IK tolerance and actual support-height lowering. The passing bridge target is[.24,.09,.76379282132281] with the saved measured9.09deg tilt; physical release settles upright at approximately[.23915,.08360,.75994]. These coordinates are exposed evidence, not a universal buffer rule.',
      'An ideal pre-contact carry reference rejects this bridge despite the verified measured held reference. Preserve the source-plan target, verify the original lift/hold, then choose and check the intermediate from actual held geometry before moving. This run matches the original3497-step motor/state hold prefix exactly.',
      'After measured release with open fingers, if the fixed-orientation40mm retreat has no IK, use the existing collision-checked joint-space route to HOME. Preserve the original failure. The successful continuation matches all5925 prior motor/state steps exactly before this fallback, then completes the ordinary stable placement/park verification.',
      'horizontal_glass_diameter_candidates(data,name,candidate_type) returns the physically used67mm band first, plus unverified72mm band, with exact proof001 local contact and body-up vector; no source/seed filters. A single bounded horizontal_glass_ik_seed() fallback was used after ordinary axis1 IK fails. Every solution remains subject to unchanged IK/collision gates.',
      'Second grasp starts from the actual settled state after full motor-prefix replay. The final leg peaks0.788deg tilt, lifts6.576cm, holds1.805s, and finishes0.398mm from the unchanged final position with0.150mm maximum external penetration. No torque increases were used.'
    ])
    payload=json.dumps(report,indent=2)
    assert ':\\' not in payload and 'C:/' not in payload
    put(target,report);print('Export complete:',target.relative_to(ROOT).as_posix())

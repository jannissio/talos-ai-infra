"""Export sanitized reach certificates with distinct geometric claim strengths."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put
source=ROOT/'.run/analytic-contact-reach-v5';out=ROOT/'docs/robotics/evidence/analytic-contact-reach-v1.json';log=ROOT/'.run/analytic-contact-reach-v1-export.log'
if out.exists() or log.exists():raise FileExistsError('Preserve evidence and log.')
preflight=require_space(out,8*1024**2)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    analysis=json.loads((source/'report.json').read_text());cases=[];hashes={}
    for record in analysis['results']:
        seed=record['seed'];folder=source/str(seed%10000)
        for name in ('report.json','scene.xml','actual-initial-state.npz','compiled-geometry-inputs.npz'):
            path=folder/name;hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        objects=[]
        for obj in record['objects']:
            item={key:value for key,value in obj.items() if key!='sides'}
            item['sides']={side:{key:value for key,value in values.items() if key not in ('primitives',)} for side,values in obj['sides'].items()}
            for values in item['sides'].values():values['classification']='analytic_no_arm_contact_at_current_pose' if values['primitive_planes_current_pose_contact_excluded'] else 'analytic_no_ordinary_two_finger_grasp_at_current_pose' if values['two_finger_grasp_excluded'] else 'unsolved_by_this_bound'
            objects.append(item)
        cases.append(dict(seed=seed,input=record['input'],input_state_sha256=record['input_state_sha256'],model_source=record['model_source'],model_xml_sha256=record['model_xml_sha256'],preserved_model_xml_identical=record['preserved_model_xml_identical'],unchanged_source_comparisons=record['unchanged_source_comparisons'],robot_bounds={side:{key:value for key,value in bounds.items() if key!='geoms'} for side,bounds in record['robot_bounds'].items()},existing_sampler_bounds=record['existing_sampler_bounds'],objects=objects))
    for path in (source/'report.json',source/'source-manifest.json',ROOT/'scripts/diagnose_analytic_contact_reach.py',Path(__file__)):
        hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(schema='analytic-contact-reach-v1',scope=analysis['scope'],preflight=preflight,derivation=analysis['derivation'],numerical_reserve_m=.00001,physical_steps=0,solver_calls=0,original_starts_changed=False,robot_model_changed=False,cases=cases,hashes_sha256=hashes,findings=[
      'A triangle-inequality ball covering every collision-enabled arm shape, including all gripper meshes, has radius500.757192mm about each fixed shoulder-pan anchor. The fixed-jaw subgroup radius is484.801354mm. These permit all hinge rotations and therefore conservatively cover every bounded-joint configuration.',
      'No arm collision surface can touch the original4001 glass,4001 spoon or4002 plate in the saved current poses. The left-arm strict support-plane gaps using a separate exact-support plane per primitive are5.758004mm,0.400845mm and23.341316mm respectively; the right-arm gaps are larger.',
      'The4002 bottle can potentially meet the left moving jaw but cannot meet any fixed-jaw geometry; the right arm cannot contact it. The same weaker ordinary-two-finger exclusion applies to4004 side_plate. Their left fixed-jaw gaps are15.691819mm and4.001824mm respectively. This does not exclude all one-jaw contact, pushing, or reorientation.',
      'At fixed object orientation, the full4mm translation allowance still leaves ordinary-two-finger separation for4001 glass/spoon and4002 plate/bottle. The4004 side_plate bound does not certify separation after4mm translation with the declared10um reserve. It must remain potentially recoverable/unsolved under such prior motion.',
      'Only4002 plate has an orientation-independent certificate surviving arbitrary rotation about its body origin and4mm body-origin translation: body-origin distance593.091726mm minus70.763691mm enclosing object radius minus500.757192mm arm radius minus4mm leaves17.570843mm.',
      'The existing sampler accepts overlapping arm and object-enclosing spheres. Its AABB-derived sphere loses directional shape and can overlap even when all true object primitives are separated. This explains false positives without claiming that every sphere overlap is feasible or that the older site-based arm radius is proven unsound.',
      'These are geometric exclusions at explicitly specified poses/motion envelopes. They are not global impossibility proofs under arbitrary cumulative relocation, object-assisted tools, prior rotations or new manipulation primitives. No original scene was discarded or resampled, and all remaining cases are unsolved by this bound rather than labeled unreachable.'
    ],preserved_incidents=[dict(run='.run/analytic-contact-reach-v1',reason='Input-audit stop: original whole-table snapshot did not include scene.py; no physics and no result classification.'),dict(run='.run/analytic-contact-reach-v2',reason='Serialization failure after first read-only geometry computation; no physics. Preserved files retained.'),dict(run='.run/analytic-contact-reach-v3',reason='Complete initial common-plane analysis preserved.'),dict(run='.run/analytic-contact-reach-v4',reason='Complete per-primitive and fixed/moving-jaw refinement preserved.')])
    payload=json.dumps(report,indent=2);assert ':\\' not in payload and 'C:/' not in payload
    put(out,report);print('Export complete.')

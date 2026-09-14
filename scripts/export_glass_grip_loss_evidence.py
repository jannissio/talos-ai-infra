"""Export sanitized, hash-bound glass contact diagnosis and one follow-up outcome."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put

target=ROOT/'docs/robotics/evidence/glass-grip-loss-diagnosis-v1.json'
log=ROOT/'.run/glass-grip-loss-diagnosis-v1-export.log'
if target.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(target,4*1024**2)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    diagnostic=ROOT/'.run/glass-grip-loss-diagnosis-v1';trial=ROOT/'.run/glass-placement-v18-proof004-band060'
    interpretation=json.loads((diagnostic/'interpretation.json').read_text())
    hashes={}
    for folder,names in [(diagnostic,('interpretation.json','contacts.json','report.json','source-manifest.json')),(trial,('result.json','states.npz','scene.xml','source-manifest.json','placement-search.json','search.json'))]:
        for name in names:
            path=folder/name;hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    diagnosis={key:interpretation[key] for key in ('scope','replay_exact','glass_weight_n','contact_friction','findings','samples','recommended_single_test','limitations')}
    result=json.loads((trial/'result.json').read_text())
    report=dict(schema='glass-grip-loss-diagnosis-v1',scope='Exposed joint seed2026114004 glass-only teacher dependency; no full-table, learned-policy or coverage claim.',storage_preflight=preflight,diagnosis=diagnosis,diagnostic_sources=json.loads((diagnostic/'source-manifest.json').read_text()),hashes_sha256=hashes,followup=dict(run=trial.relative_to(ROOT).as_posix(),contact_band_m=.060,torque_nm=.65,changes='Only contact band changed from69mm to60mm; same local tool point/local axis/opening. Additional rigid-route preflight required before new manipulation.',planning=json.loads((trial/'placement-search.json').read_text()),result=result,interpretation='The deeper band enabled6.0555cm lift and1.805s unsupported hold, supporting the lip-headroom diagnosis. The glass still escaped at the beginning of align; final placement remains unsolved. This is not a verified full-transfer grasp.',source_hashes=json.loads((trial/'source-manifest.json').read_text())),physical_trials_added=1,model_or_physics_changes=False,next_trial_authorized=False)
    payload=json.dumps(report,indent=2)
    assert ':\\' not in payload and 'C:/' not in payload and '/Users/' not in payload
    put(target,report);print('Exported',target.relative_to(ROOT).as_posix())

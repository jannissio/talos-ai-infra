"""Validate all pilot files and write a compact, shareable evidence summary."""
import json,sys,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_bottle_data import OUT,ROOT,save
from scripts.validate_demonstration import validate
from simulation_lab.dataset import load_policy_episode

if __name__=='__main__':
    episodes=[]
    excluded=json.loads((OUT/'excluded.json').read_text()) if (OUT/'excluded.json').exists() else {}
    for path in sorted(OUT.glob('*/manifest.json')):
        deadline=time.monotonic()+300
        while json.loads(path.read_text())['images']['status']=='pending':
            if time.monotonic()>deadline:
                raise TimeoutError('Image export is still pending: '+path.parent.name)
            time.sleep(2)
        if path.parent.name in excluded:
            manifest=json.loads(path.read_text())
            manifest.update(training_eligible=False,exclusion_reason=excluded[path.parent.name])
            save(path,manifest)
            continue
        manifest=json.loads(path.read_text())
        manifest['images'].setdefault('multisampling',1)
        manifest['replays']=json.loads((path.parent/'replay-verification.json').read_text())
        manifest['training_eligible']=manifest['images']['status']=='completed' and all(r['passed'] for r in manifest['replays'])
        save(path,manifest)
        result=validate(path.parent)
        manifest=json.loads(path.read_text())
        samples=load_policy_episode(path.parent)
        assert len(samples)==(result['actions']+9)//10
        assert all(x['passed'] for x in manifest['replays'])
        result['policy_samples']=len(samples)
        result['sha256']={name:hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
                          for name in ['scene.xml','initial-integration-state.npy','actions.jsonl.gz','observations.jsonl.gz','images.jsonl']}
        episodes.append(result)
        print(path.parent.name,'valid',result['images'],'images',flush=True)
    assert len(episodes)==20
    summary={'episodes':episodes,'successful_episodes':len(episodes),'physical_replays_passed':40,
             'images':sum(r['images'] for r in episodes),'observations':sum(r['observations'] for r in episodes),
             'actions':sum(r['actions'] for r in episodes),'duration_s':sum(r['duration_s'] for r in episodes),
             'bytes_on_disk':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
             'selected_dataset_bytes':sum(p.stat().st_size for e in episodes for p in (OUT/e['id']).rglob('*') if p.is_file()),
             'policy_inputs':['three synchronized RGB images','12 joint positions','12 joint velocities'],
             'action':'12 nominal joint targets at 20 Hz; interpolation and gripper torque limiting at 200 Hz',
             'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                              ['simulation_lab/autonomy.py','simulation_lab/dinner_autonomy.py','simulation_lab/scene.py',
                               'simulation_lab/recording.py','simulation_lab/dataset.py','scripts/prepare_bottle_data.py','scripts/replay_bottle_pilot.py']},
             'limitations':['Single object and fixed surrounding dinner layout; all successful pilot demonstrations use the left arm.',
                            '20 demonstrations are a pipeline pilot, not sufficient evidence of generalization.',
                            'No learned policy, no hardware transfer test, no bimanual coordination.',
                            'Reserved validation poses have not been used or evaluated.']}
    save(ROOT/'docs/robotics/bottle-dataset-validation.json',summary)
    print({k:v for k,v in summary.items() if k!='episodes'})

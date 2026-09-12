"""Convert clean pilot episodes to local LeRobot v3 datasets, with whole-episode splits."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulation_lab.dataset import load_policy_episode
from simulation_lab.storage import require_space,GIB

CAMERAS=['overhead','left_wrist_cam','right_wrist_cam']
VALIDATION=['sideways-05','sideways-08','upright-08','upright-10']
FIT=['sideways-01','sideways-03','sideways-11','upright-01','upright-04']

def convert(source,output,split,width=320,height=240,wait_for_fit5=False):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite dataset: '+str(output))
    excluded=json.loads((source/'excluded.json').read_text()) if (source/'excluded.json').exists() else {}
    names=[p.parent.name for p in sorted(source.glob('*/manifest.json')) if p.parent.name not in excluded and json.loads(p.read_text()).get('training_eligible')]
    names=[n for n in names if n in FIT] if split=='fit5' else [n for n in names if (n in VALIDATION)==(split=='validation')]
    if wait_for_fit5:
        if split!='fit5':raise ValueError('Waiting is supported only for the explicit five-episode fit set.')
        names=FIT.copy()
    if not names:raise ValueError('No eligible episodes.')
    require_space(output,int(GIB))
    features={'observation.state':{'dtype':'float32','shape':(24,),'names':[f'q{i}' for i in range(12)]+[f'dq{i}' for i in range(12)]},
              'action':{'dtype':'float32','shape':(12,),'names':[f'target{i}' for i in range(12)]}}
    for camera in CAMERAS:features['observation.images.'+camera]={'dtype':'image','shape':(height,width,3),'names':['height','width','channels']}
    dataset=LeRobotDataset.create(repo_id='talos/bottle-'+split,root=output,fps=20,features=features,
                                  robot_type='dual_so101_simulation',use_videos=False,video_backend='pyav',image_writer_threads=4)
    lineage=[]
    try:
        for name in names:
            if wait_for_fit5:
                deadline=time.monotonic()+1200
                while True:
                    try:
                        ready=json.loads((source/name/'manifest.json').read_text())
                        if ready.get('training_eligible') and ready.get('images',{}).get('status')=='completed':break
                    except (FileNotFoundError,json.JSONDecodeError):pass
                    if time.monotonic()>deadline:raise TimeoutError('Waiting for completed capture: '+name)
                    print('Waiting for capture',name,flush=True);time.sleep(10)
            source_meta=json.loads((source/name/'manifest.json').read_text())
            require_space(output,max(int(.25*GIB),int(source_meta['observation_count']*width*height*3*1.2)))
            samples=load_policy_episode(source/name)
            for sample in samples:
                frame={'observation.state':np.concatenate([sample['joint_position'],sample['joint_velocity']]),
                       'action':sample['action'],'task':'Place the bottle upright in the serving area.'}
                for camera in CAMERAS:
                    with Image.open(sample['images'][camera]) as image:
                        frame['observation.images.'+camera]=np.asarray(image.convert('RGB').resize((width,height),Image.Resampling.BILINEAR))
                dataset.add_frame(frame)
            dataset.save_episode()
            lineage.append({'episode_index':len(lineage),'source':name,'frames':len(samples),
                            'gripper_cap_nm':source_meta.get('gripper_torque_cap_nm'),
                            'matched_controller':source_meta.get('matched_controller'),
                            'source_manifest_sha256':hashlib.sha256((source/name/'manifest.json').read_bytes()).hexdigest()})
            print(split,name,len(samples),flush=True)
    finally:dataset.finalize()
    (output/'talos_lineage.json').write_text(json.dumps({'split':split,'episodes':lineage,'cameras':CAMERAS,
        'gripper_caps_nm':sorted(set(x['gripper_cap_nm'] for x in lineage if x['gripper_cap_nm'] is not None)),
        'resolution':[width,height],'state_order':'12 joint positions followed by 12 joint velocities',
        'limitations':'Validation episodes are nearby development poses; reserved poses are separate and unconsumed.'},indent=2))
    # Read the actual on-disk artifact, rather than trusting the writer alone.
    check=LeRobotDataset(repo_id='talos/bottle-'+split,root=output,video_backend='pyav')
    assert len(check)==sum(r['frames'] for r in lineage)
    row=check[0]
    assert row['observation.state'].shape==(24,) and row['action'].shape==(12,)
    print('Verified',len(check),'frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='.run/bottle-pilot');p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['fit5','train','validation'],default='fit5');p.add_argument('--wait-for-fit5',action='store_true');a=p.parse_args()
    convert(a.source,a.output,a.split,wait_for_fit5=a.wait_for_fit5)

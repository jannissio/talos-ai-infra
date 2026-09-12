"""Independently reconstruct matched physics and compare every saved robot state."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image
from scripts.prepare_bottle_data import STATE,save
from simulation_lab.policy_control import apply_targets

def audit(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(folder/'initial-integration-state.npy',allow_pickle=False),STATE)
    mujoco.mj_forward(model,data)
    with gzip.open(folder/'observations.jsonl.gz','rt') as f:obs={r['action_index']:r for r in map(json.loads,f)}
    with gzip.open(folder/'actions.jsonl.gz','rt') as f:actions=[json.loads(s) for s in f]
    maximum=0.;control_error=0.
    model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(model,height=240,width=320);option=mujoco.MjvOption();option.geomgroup[3:]=0
    sample_indices={0,1000,5000,(len(actions)-1)//10*10};pixel_error=0;image_checks=0
    for i,row in enumerate(actions):
        if i in obs:
            maximum=max(maximum,float(np.max(np.abs(data.qpos-np.array(obs[i]['qpos'])))),float(np.max(np.abs(data.qvel-np.array(obs[i]['qvel'])))))
        if i in sample_indices:
            for camera in ['overhead','left_wrist_cam','right_wrist_cam']:
                renderer.update_scene(data,camera=camera,scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                actual=renderer.render().astype(int)
                expected=np.asarray(Image.open(folder/'images'/f'{obs[i]["index"]:06d}_{camera}.png')).astype(int)
                pixel_error=max(pixel_error,int(np.max(np.abs(actual-expected))));image_checks+=1
        control=apply_targets(model,data,row['target'])
        control_error=max(control_error,float(np.max(np.abs(control-np.array(row['ctrl'])))))
        data.ctrl[:]=control;mujoco.mj_step(model,data)
    renderer.close()
    terminal=obs[len(actions)]
    maximum=max(maximum,float(np.max(np.abs(data.qpos-np.array(terminal['qpos'])))),float(np.max(np.abs(data.qvel-np.array(terminal['qvel'])))))
    assert maximum<1e-10 and control_error<1e-10 and pixel_error==0,(folder.name,maximum,control_error,pixel_error)
    assert all(r['passed'] for r in manifest['replays'])
    # Preserve manifest bytes used by any concurrent conversion: this is an audit,
    # not an in-place rewrite of source provenance.
    return {'episode':folder.name,'max_state_error':maximum,'max_applied_control_error':control_error,
            'max_pixel_error':pixel_error,'image_checks':image_checks,
            'nonterminal_observations':len(obs)-1,'physical_replays_passed':len(manifest['replays'])}

if __name__=='__main__':
    import torch
    torch.cuda.init()
    root=Path('.run/bottle-matched-nvidia')
    rows=[audit(p.parent) for p in sorted(root.glob('*/manifest.json'))]
    assert len(rows)==5
    save(Path('docs/robotics/act-matched-reconstruction.json'),rows);print(json.dumps(rows,indent=2))

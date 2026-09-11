"""Render documented reset examples; these images are not task successes."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
from PIL import Image
from scripts.evaluate_dinner_scene import load
from simulation_lab.scene import build_scene

root=Path(__file__).resolve().parents[1]/'docs/robotics'
records=[]
for preset,opened,name in [('task',True,'dinner-task.jpg'),('reference',False,'dinner-reference.jpg')]:
    m,d,l=load(42,preset,opened)
    for _ in range(200):mujoco.mj_step(m,d)
    with mujoco.Renderer(m,height=540,width=960) as renderer:
        renderer.update_scene(d,camera='center')
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        Image.fromarray(renderer.render()).save(root/name,quality=90)
    xml,_=build_scene(seed=42,scenario='dinner',dinner_preset=preset,drawer_open=opened)
    records.append({'file':name,'preset':preset,'drawer_open':opened,'camera':'center','seed':42,
                    'simulation_time_s':float(d.time),'source':'actual MuJoCo renderer',
                    'autonomous_result':False,'xml_sha256':hashlib.sha256(xml.encode()).hexdigest()})
(root/'dinner-images.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print('Rendered two reset examples; neither is an autonomous result.')

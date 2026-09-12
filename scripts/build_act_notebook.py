"""Generate the reviewable notebook corresponding to the first Colab experiment."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build():
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('''# Talos: first ACT bottle learning experiment
This is a **200-update pipeline diagnostic**, not a trained competition solution or a generalization result.
Use a free T4 runtime. The five familiar episodes contain 4,255 observations: overhead + both wrist RGB cameras at 320×240, 12 joint positions and 12 velocities. ACT predicts 20 future joint-target vectors. Physics runs at 200 Hz; observations at 20 Hz; replan every four intervals.

The simulator teacher used privileged positions to make demonstrations. Those positions, contact forces and task stages are excluded from the learned policy inputs. Evaluation observes contacts separately. There are no welds, teleports or teacher corrections during policy rollout.

Run `scripts/package_act_colab.py` locally, then upload the resulting task package below. Nothing is published to Hugging Face. Runtime files are temporary: download the result archive before disconnecting. No Drive-wide permission is required.
''')
    code('''import sys, subprocess, pathlib
print(sys.version)
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True))
with open('/content/talos-install.log','w') as log:
    subprocess.run(['apt-get','update','-qq'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['apt-get','install','-y','-qq','python3.12-venv'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python3.12','-m','venv','/content/talos-env'],stdout=log,stderr=subprocess.STDOUT,check=True)
    PY='/content/talos-env/bin/python'
    r=subprocess.run([PY,'-m','pip','install','lerobot[training]==0.6.1','torch==2.8.0','torchvision==0.23.0','torchcodec==0.7.0','numpy==2.2.6','mujoco==3.12.0','PyOpenGL==3.1.10'],stdout=log,stderr=subprocess.STDOUT)
print(pathlib.Path('/content/talos-install.log').read_text()[-4000:])
assert r.returncode==0
''')
    code('''from google.colab import files
# Alternatively upload through Colab's Files panel and wait until upload completes.
files.upload()
''')
    code('''import hashlib, zipfile
package=pathlib.Path('/content/talos-act-colab.zip')
# Paste the SHA256 from your local .run/talos-act-colab.zip.json if rebuilding the package.
EXPECTED_SHA256='98c1f276eb73e46e6b9eed1b46a4347f5be4bd813f6fc51219850f52cf87e5ed'
assert hashlib.sha256(package.read_bytes()).hexdigest()==EXPECTED_SHA256
ROOT=pathlib.Path('/content/talos');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(package) as z:
    assert all((ROOT/n).resolve().is_relative_to(ROOT) for n in z.namelist())
    z.extractall(ROOT)
print('Verified and extracted task package')
''')
    code('''with open('/content/talos-train.log','w') as log:
    result=subprocess.run([PY,'scripts/train_act.py','--dataset','.run/act-fit5','--output','.run/act-colab-200','--steps','200','--batch-size','8','--device','cuda','--log-every','20','--save-every','200'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
print(pathlib.Path('/content/talos-train.log').read_text()[-8000:])
assert result.returncode==0
''')
    md('''## Optional cloud physical evaluation
Local evaluation after downloading the checkpoint is recommended: the measured Colab trial took about 230 seconds for 60 simulated seconds. The cell below runs one familiar start; run the remaining starts locally with `scripts/evaluate_act.py`.
Lower training loss does not establish task success. These are familiar-start evaluations. The fixed 0.25 Nm gripper cap passed teacher-action replay on these five starts, but failed on two other upright pilot starts. The original teacher used different caps for upright/sideways grasps. Resolve that action-contract mismatch before claiming a 20-episode policy benchmark.

Evaluation runs synchronously offline: simulation waits during image inference. Report measured latency separately from the nominal control frequency. A successful outcome needs physical lift, supported hold, stable upright release near the destination, and limits on other-arm/object movement.
''')
    code('''import os
env=dict(os.environ,MUJOCO_GL='egl')
episodes=['upright-01']
for name in episodes:
    output='.run/act-colab-200/eval-'+name+'.json'
    r=subprocess.run([PY,'scripts/evaluate_act.py','--checkpoint','.run/act-colab-200/step-000200','--episode','.run/bottle-pilot/'+name,'--output',output,'--seconds','60','--gripper-cap','.25'],cwd=ROOT,env=env,capture_output=True,text=True)
    print(r.stdout,r.stderr)
    assert r.returncode==0
''')
    md('''## Save before disconnecting
The checkpoint contains model weights, normalization, configuration, dataset lineage, optimizer and scaler state. `--resume CHECKPOINT --steps NEW_TOTAL` continues updates; shuffled data iteration restarts, so continuation is not bitwise identical. Do not load optimizer files from unknown sources.
''')
    code('''import shutil
archive=shutil.make_archive('/content/talos-act-result','zip',ROOT/'.run/act-colab-200')
files.download(archive)
''')
    package_report=ROOT/'.run/talos-act-colab.zip.json'
    if package_report.exists():
        sha=json.loads(package_report.read_text())['sha256']
        for cell in cells:
            cell['source']=[s.replace('98c1f276eb73e46e6b9eed1b46a4347f5be4bd813f6fc51219850f52cf87e5ed',sha) for s in cell['source']]
    out=ROOT/'notebooks/Talos_ACT_Bottle_Pilot.ipynb';out.parent.mkdir(exist_ok=True)
    for index,cell in enumerate(cells):cell['id']=f'talos-{index:02d}'
    out.write_text(json.dumps({'cells':cells,'metadata':{'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5},indent=2),encoding='utf-8')
    for cell in cells:
        if cell['cell_type']=='code':compile(''.join(cell['source']),str(out),'exec')
    print(out)
if __name__=='__main__':build()

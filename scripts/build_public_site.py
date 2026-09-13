"""Build a static evidence page using an explicit public-file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space

ROOT=Path(__file__).resolve().parents[1]
ASSETS={
    'cover-v6.png': 'submission/final-v6/cover-v6.png',
    'Talos-demo-v6.mp4': 'submission/final-v6/Talos-demo-v6.mp4',
    'Talos-dinner-ten-v6.mp4': 'submission/final-v6/Talos-dinner-ten-v6.mp4',
    'Talos-relays-ten-v3.mp4': 'submission/final-v6/Talos-relays-ten-v3.mp4',
    'Talos-composed-dinner-ten-v1.mp4': 'submission/final-v6/Talos-composed-dinner-ten-v1.mp4',
    'rgb-feedback-experiment.mp4': 'submission/final-v6/rgb-feedback-experiment.mp4',
    'Talos-v6.pdf': 'submission/final-v6/Talos-v6.pdf',
    'evidence/dinner-summary.json': 'docs/robotics/evidence/learned-dinner-v6/summary.json',
    'evidence/relay-summary.json': 'docs/robotics/evidence/learned-relays-v3/summary.json',
    'evidence/composed-summary.json': 'docs/robotics/evidence/composed-dinner-v1/original-final-summary.json',
    'licenses/Talos-MIT.txt': 'LICENSE',
    'licenses/SO101-NOTICE.md': 'simulation_lab/NOTICE.md',
    'licenses/SO101-Apache-2.0.txt': 'simulation_lab/assets/so101/LICENSE',
}


def build(output):
    if output.exists():raise FileExistsError('Choose a fresh build directory; previous exports are preserved.')
    files=[(ROOT/'site'/name,Path(name)) for name in ('index.html','style.css','app.js')]
    files += [(ROOT/source,Path('assets')/name) for name,source in ASSETS.items()]
    require_space(output,sum(source.stat().st_size for source,_ in files)+16*1024**2)
    output.mkdir(parents=True)
    manifest={}
    for source,relative in files:
        require_space(output,source.stat().st_size+1024**2)
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        manifest[relative.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    (output/'.nojekyll').write_text('')
    (output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':output.as_posix(),'files':len(files),'bytes':sum((output/p).stat().st_size for p in manifest)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)

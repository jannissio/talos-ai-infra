"""Package only task code, embedded-image pilot data and five physical reset scenes."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space

def package():
    output=ROOT/'.run/talos-act-colab.zip';paths=[]
    for directory in ['simulation_lab','scripts']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',))
    paths.append(ROOT/'requirements-training.txt')
    dataset=ROOT/'.run/act-fit5'
    paths.extend(p for p in dataset.rglob('*') if p.is_file() and 'images' not in p.relative_to(dataset).parts)
    lineage=json.loads((dataset/'talos_lineage.json').read_text())
    for row in lineage['episodes']:
        folder=ROOT/'.run/bottle-pilot'/row['source']
        paths.extend(folder/n for n in ['scene.xml','manifest.json','initial-integration-state.npy'])
    paths=sorted(set(paths))
    require_space(output,int(sum(path.stat().st_size for path in paths)*1.05)+32*1024**2)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in paths:
            require_space(output,int(path.stat().st_size*1.05)+8*1024**2)
            z.write(path,path.relative_to(ROOT).as_posix())
    sha=hashlib.sha256(output.read_bytes()).hexdigest()
    report={'path':str(output),'bytes':output.stat().st_size,'sha256':sha,'files':len(set(paths)),
            'contains':'Task source, SO-101 assets, embedded-image fit5 dataset, five reset scenes. No account files or credentials.'}
    (output.with_suffix('.zip.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':package()

"""Package the verified pilot for extraction into a checkout on Colab/home PC."""
import hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_bottle_data import OUT,ROOT,save
from simulation_lab.storage import require_space

if __name__=='__main__':
    report=ROOT/'docs/robotics/bottle-dataset-validation.json'
    summary=json.loads(report.read_text())
    assert summary['successful_episodes']==20
    archive=ROOT/'.run/bottle-pilot.zip'
    files=[file for episode in summary['episodes'] for file in (OUT/episode['id']).rglob('*') if file.is_file()]
    files += [OUT/'reserved-validation.json',report,ROOT/'docs/robotics/bottle-camera-calibration.json',ROOT/'docs/robotics/DATA_PREPARATION.md']
    require_space(archive,sum(file.stat().st_size for file in files)+32*1024**2)
    # Refuse to replace an existing export silently.
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as output:
        for file in files:
            require_space(archive,file.stat().st_size+8*1024**2)
            output.write(file,file.relative_to(ROOT).as_posix())
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    save(archive.with_suffix('.zip.json'),{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest.hexdigest(),
         'instructions':'Extract at the root of a matching Talos repository checkout. SO-101 assets and source code are supplied by that checkout, not duplicated here.'})
    print(archive)

"""Apply the final physical replay checks independently to the selected pilot."""
import json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_bottle_data import OUT,ROOT,save
from scripts.replay_bottle_pilot import replay_saved

def check(folder):
    results=[replay_saved(folder,hz) for hz in (200,20)]
    save(folder/'replay-verification.json',results)
    return {'id':folder.name,'replays':results}

if __name__=='__main__':
    excluded=json.loads((OUT/'excluded.json').read_text())
    folders=[p.parent for p in sorted(OUT.glob('*/manifest.json')) if p.parent.name not in excluded]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(check,folders))
    save(ROOT/'docs/robotics/bottle-replay-audit.json',results)
    print([(r['id'],[a['passed'] for a in r['replays']]) for r in results])
    assert len(results)==20 and all(a['passed'] for r in results for a in r['replays'])

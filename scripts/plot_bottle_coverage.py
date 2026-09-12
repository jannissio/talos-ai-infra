"""Draw an aggregate coverage map without treating IK failure as unreachable."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_bottle_data import ROOT

if __name__=='__main__':
    rows=json.loads((ROOT/'docs/robotics/bottle-workspace.json').read_text())['samples']
    pilot=json.loads((ROOT/'docs/robotics/bottle-pilot-results.json').read_text())['attempts']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="510" viewBox="0 0 1000 510">',
           '<rect width="1000" height="510" fill="#14212e"/>',
           '<g font-family="sans-serif" fill="#e1edf5"><text x="30" y="30" font-size="22">Bottle coverage: approach candidates and completed pilot motions</text>',
           '<text x="30" y="55" font-size="14">Each grid square aggregates both arms and sampled headings. It does not certify full task reachability.</text>']
    for panel,mode in enumerate([False,True]):
        left=70+panel*490
        parts.append(f'<text x="{left}" y="95" font-size="18">{"Sideways" if mode else "Upright"} bottle starts</text>')
        for ix in range(7):
            x=round(-.15+ix*.05,4)
            for iy in range(5):
                y=round(-.20+iy*.05,4)
                group=[r for r in rows if r['sideways']==mode and abs(r['x']-x)<1e-5 and abs(r['y']-y)<1e-5]
                candidate=any(r['label']=='approach_candidate' for r in group)
                overlap=all(r['label']=='reset_overlap' for r in group)
                color='#23796f' if candidate else '#714d45' if overlap else '#465665'
                px=left+ix*48;py=120+(4-iy)*48
                parts.append(f'<rect x="{px}" y="{py}" width="44" height="44" fill="{color}"/>')
            parts.append(f'<text x="{left+ix*48}" y="385" font-size="12">{x*100:.0f}</text>')
        for trial in pilot:
            spec=trial['spec']
            if spec['sideways']!=mode:continue
            px=left+22+(spec['x']+.15)/.05*48;py=142+(-spec['y'])/.05*48
            parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#ffce71" stroke="#14212e" stroke-width="1"/>')
        for iy in range(5):
            parts.append(f'<text x="{left-32}" y="{148+iy*48}" font-size="12">{-iy*5}</text>')
        parts.append(f'<text x="{left+110}" y="408" font-size="13">X (cm); Y labels at left (cm)</text>')
    parts.extend(['<text x="30" y="450" font-size="14">Teal: at least one approach candidate · Grey: unresolved/planned-path collision · Brown: all tested resets overlap</text>',
                  '<text x="30" y="477" font-size="14" fill="#ffce71">Gold dots: 20 completed physical pilot episodes (some points overlap). Three validation poses remain reserved.</text></g></svg>'])
    (ROOT/'docs/robotics/bottle-coverage.svg').write_text('\n'.join(parts),encoding='utf-8')

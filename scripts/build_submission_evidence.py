"""Copy selected credential-free metrics into public submission evidence."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space

def build():
    source=Path('.run/submission-work');out=Path('submission/evidence');require_space(out,8*1024**2);out.mkdir(parents=True,exist_ok=True)
    def read(name):return json.loads((source/name).read_text())
    files={'programmed-table.json':read('voice-set-table/result.json'),'programmed-relay.json':read('command-relay-right/result.json'),
           'learned-bottle.json':read('voice-neural-bottle/result.json'),'intel-network-benchmark.json':read('primitive-openvino/benchmark.json'),
           'ten-seed-bottle.json':read('ten-seed-bottle-v1/summary.json'),
           'upright-visual-bottle.json':read('voice-visual-bottle/result.json'),
           'ten-seed-upright-visual.json':read('ten-seed-visual-bottle-v1/summary.json'),
           'upright-visual-development.json':read('visual400-development/summary.json'),
           'upright-visual-network-benchmark.json':read('visual-bottle-openvino-clean/benchmark.json'),
           'upright-visual-blank.json':read('visual-blank-check.json'),
           'upright-visual-cancel.json':read('visual-cancel-check.json'),
           'upright-visual-collision-regression.json':read('visual-collision-regression/summary.json')}
    # These historical learned runs predate the actual collision monitor.
    # Its old, initialized counter was not a measurement; preserve no false zero.
    def strip_unmeasured(value):
        if isinstance(value,dict):
            value.pop('unexpected_collisions',None)
            for child in value.values():strip_unmeasured(child)
        elif isinstance(value,list):
            for child in value:strip_unmeasured(child)
    for name in ['learned-bottle.json','ten-seed-bottle.json','upright-visual-bottle.json',
                 'ten-seed-upright-visual.json','upright-visual-development.json',
                 'upright-visual-blank.json','upright-visual-cancel.json']:
        strip_unmeasured(files[name])
        files[name]['measurement_note']='Historical run before explicit unexpected-arm-contact counting. The unmeasured counter is omitted. upright-visual-collision-regression.json reports a later 10/10 regression on the same exposed seeds with actual contact checks; it is not a fresh holdout.'
    speech=[]
    for name in ['speechmatics-table-test.json','speechmatics-bottle-test.json']:
        r=read(name);speech.append({k:r[k] for k in ['provider','audio_source','audio_seconds','wall_seconds','transcript','credential_or_token_logged']})
    files['speechmatics.json']={'tests':speech,'execution_evidence':['programmed-table.json','learned-bottle.json','upright-visual-bottle.json'],
        'note':'The bottle transcript initially failed parsing due to sentence punctuation. A constrained punctuation fix and successful physical execution are recorded in learned-bottle.json. Synthetic audio, not a human microphone test.'}
    for name,value in files.items():(out/name).write_text(json.dumps(value,indent=2)+'\n')
    print('Wrote',len(files),'selected evidence files.')

if __name__=='__main__':build()

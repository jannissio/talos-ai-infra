"""Loopback-only, isolated physical policy trials; does not control the live lab server."""
import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space

EPISODES = ['upright-01', 'upright-04', 'sideways-01', 'sideways-03', 'sideways-11']
PAGE = '''<!doctype html><meta charset="utf-8"><title>Talos · Bottle policy</title>
<style>body{font:17px system-ui;background:#101d2a;color:#e4edf6;max-width:950px;margin:36px auto;padding:0 24px}h1{font-size:32px}p{line-height:1.6;color:#b8cad9}select,button{font:inherit;padding:10px;margin:6px 8px 6px 0;border-radius:7px;border:1px solid #496578;background:#1b3345;color:white}button{cursor:pointer}button:disabled{opacity:.5}video{width:100%;max-height:540px;background:#08111a;border-radius:12px}#status{padding:16px;background:#203849;border-radius:8px;margin:18px 0}#detail{white-space:pre-wrap;font-size:15px}a{color:#72ded6}</style>
<h1>Talos · Learned bottle baseline</h1>
<p>Camera and joint feedback choose movements from fitted demonstrations. This is a nonparametric retrieval model, <b>not ACT or a language model</b>. Each run starts a separate MuJoCo scene, physically lifts the bottle and places it upright. No teacher corrections, attachments or teleports.</p>
<label>Start <select id="episode"></select></label><select id="fault"><option value="none">Normal trial</option><option value="stall">Two-second actuator pause during approach</option><option value="blank">Blank cameras (should reject)</option></select><button id="run">Run policy</button><button id="cancel" disabled>Cancel</button>
<div id="status">Loading results…</div><p>Recording from the last completed trial; updates when the next trial finishes.</p><video id="video" controls muted playsinline></video><p id="detail"></p>
<p>These five starts appeared in training. They establish a limited working baseline, not arbitrary-position generalization. Evaluation waits for inference; timing is simulated time, not a real-time performance claim. The main simulator remains at <a href="http://127.0.0.1:8765/">port 8765</a>.</p>
<script>
const el=id=>document.getElementById(id);let lastVideo='';
for(const name of ['upright-01','upright-04','sideways-01','sideways-03','sideways-11'])el('episode').add(new Option(name,name));
function show(r){if(!r){el('detail').textContent='';return}el('detail').textContent=(r.success?'PASS: ':'Result: ')+r.reason+'\\nLift: '+r.max_lift_cm.toFixed(2)+' cm · placement error: '+(r.placement_error_m*1000).toFixed(2)+' mm · simulated time: '+r.simulated_s.toFixed(2)+' s';if(r.video_url&&r.video_url!==lastVideo){lastVideo=r.video_url;el('video').src=r.video_url}}
async function poll(){try{const s=await(await fetch('/state')).json();el('run').disabled=s.running;el('cancel').disabled=!s.running;el('status').textContent=s.message;show(s.result)}catch(e){el('status').textContent='Local demo server unavailable: '+e.message}}
async function command(path,data){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error(await r.text());await poll()}
el('run').onclick=()=>command('/run',{episode:el('episode').value,fault:el('fault').value}).catch(e=>el('status').textContent=e.message);
el('cancel').onclick=()=>command('/cancel',{}).catch(e=>el('status').textContent=e.message);
poll();setInterval(poll,1000);
</script>'''


def main(args):
    checkpoint = Path(args.checkpoint).resolve()
    primitive = (checkpoint / 'primitive.safetensors').is_file()
    neural = primitive or (checkpoint / 'sequence.safetensors').is_file()
    if not neural and not (checkpoint / 'retrieval.npz').is_file():
        raise ValueError('This demo requires a fitted retrieval or recurrent neural checkpoint.')
    episodes = EPISODES
    page = PAGE
    if neural:
        meta = json.loads((checkpoint / ('primitive.json' if primitive else 'sequence.json')).read_text())
        episodes = [name for name in EPISODES if name in meta['training_episodes']]
        if not episodes:
            raise ValueError('No supported demonstration starts in this checkpoint.')
        page = page.replace('Camera and joint feedback choose movements from fitted demonstrations. This is a nonparametric retrieval model, <b>not ACT or a language model</b>.',
            'A two-layer recurrent neural network predicts joint commands from three camera images and measured arm joints. It uses fixed PCA image features; it is <b>not ACT or a language model</b>. Demonstration actions are absent at inference.')
        page = page.replace("['upright-01','upright-04','sideways-01','sideways-03','sideways-11']", json.dumps(episodes))
        page = page.replace('These five starts appeared in training.', 'The listed starts appeared in training. A normal-trial success does not establish pause recovery; fault controls are diagnostic tests.')
        page = page.replace('Talos · Learned bottle baseline', 'Talos · Neural bottle experiment')
        if meta.get('tracking_replay_chunk'):
            page = page.replace('Demonstration actions are absent at inference.',
                'Demonstration actions are absent at inference. A programmed joint-feedback guard repeats the last predicted segment if the arm lags, and stops after a persistent stall. Short-pause recovery is a property of this combined controller.')
        if primitive:
            page = page.replace('A two-layer recurrent neural network predicts joint commands from three camera images and measured arm joints. It uses fixed PCA image features; it is <b>not ACT or a language model</b>. Demonstration actions are absent at inference.',
                'A neural network generates a motion trajectory conditioned on the initial three camera images. Joint feedback pauses progress when the arm lags. This is a <b>learned motion primitive</b>, not continuous visual correction, ACT or a language model. A programmed guard repeats the last predicted segment and stops after a persistent stall. Demonstration actions are absent at inference.')
    output = ROOT / '.run' / ('bottle-sequence-demo' if neural else 'bottle-policy-demo')
    output.mkdir(exist_ok=True)
    lock = threading.RLock()
    state = {'running': False, 'message': 'Ready to run a physical trial.', 'result': None, 'cancel_requested': False}
    process = [None]
    media = {}

    def present(path):
        row = json.loads(path.read_text())
        result = {k: row[k] for k in ['success', 'reason', 'max_lift_cm', 'placement_error_m', 'simulated_s']}
        if row.get('video') and Path(row['video']).is_file():
            key = uuid.uuid4().hex
            media[key] = Path(row['video'])
            result['video_url'] = '/video/' + key
        return result

    evidence = Path(args.evidence)
    if evidence.is_dir():
        evidence = evidence / 'upright-01.json'
    if evidence.is_file():
        state['result'] = present(evidence)
        state['message'] = 'Verified saved trial shown below. Run policy to repeat it.'

    def trial(episode, fault):
        folder = output / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
        try:
            require_space(output, 20 * 1024**2)
            folder.mkdir()
            result = folder / 'result.json'
            command = [sys.executable, str(ROOT / 'scripts/evaluate_act.py'), '--checkpoint', str(checkpoint),
                       '--episode', str(ROOT / '.run/bottle-matched-nvidia' / episode), '--device', 'cpu',
                       '--output', str(result), '--video', str(folder / 'trial.mp4')]
            if fault == 'blank':
                command += ['--blank-images']
            if fault == 'stall':
                command += ['--stall-start-seconds', '2', '--stall-at-seconds', '3']
            with (folder / 'evaluation.log').open('w') as log:
                with lock:
                    if state['cancel_requested']:
                        raise RuntimeError('Trial cancelled before launch.')
                    process[0] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    child = process[0]
                try:
                    code = child.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    stop_child(child)
                    raise RuntimeError('Trial exceeded the five-minute wall-time limit.')
            with lock:
                if code != 0:
                    state['message'] = 'Trial cancelled or failed to execute. See ' + str(folder / 'evaluation.log')
                else:
                    state['result'] = present(result)
                    state['message'] = ('Physical checks passed.' if state['result']['success'] else 'Trial finished: ' + state['result']['reason'])
        except Exception as exc:
            with lock:
                state['message'] = 'Trial stopped: ' + str(exc)
        finally:
            with lock:
                process[0] = None
                state['running'] = False

    def stop_child(child):
        if child and child.poll() is None:
            # Windows venv's redirector has a child Python process. Terminate only this owned tree.
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=15)
            else:
                child.terminate()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload, mime='application/json'):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                return self.reply(200, page.encode(), 'text/html; charset=utf-8')
            if self.path == '/state':
                with lock:
                    return self.reply(200, dict(state))
            if self.path.startswith('/video/'):
                with lock:
                    path = media.get(self.path.removeprefix('/video/'))
                if path:
                    return self.reply(200, path.read_bytes(), 'video/mp4')
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if host not in [f'127.0.0.1:{args.port}', f'localhost:{args.port}'] or (origin and urlparse(origin).netloc != host):
                return self.reply(403, {'error': 'Loopback same-origin requests only.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 1024:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length) or b'{}')
                with lock:
                    if self.path == '/cancel':
                        state['cancel_requested'] = True
                        stop_child(process[0])
                        return self.reply(200, {'ok': True})
                    if self.path != '/run':
                        return self.reply(404, {'error': 'Not found'})
                    episode, fault = request.get('episode'), request.get('fault', 'none')
                    if episode not in episodes or fault not in ['none', 'blank', 'stall']:
                        raise ValueError('Choose a listed start and fault mode.')
                    if state['running']:
                        return self.reply(409, {'error': 'A trial is already running.'})
                    state.update(running=True, message='Running ' + episode + ' in a separate physical scene…', result=None, cancel_requested=False)
                    threading.Thread(target=trial, args=(episode, fault), daemon=True).start()
                    return self.reply(202, {'ok': True})
            except (ValueError, TypeError, AttributeError) as exc:
                self.reply(400, {'error': str(exc)})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Bottle policy demo: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    finally:
        stop_child(process[0])
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--port', type=int, default=8766)
    main(parser.parse_args())

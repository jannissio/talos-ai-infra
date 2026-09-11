"""Exercise a running local sandbox. Resets physics; restores layout/control settings."""
from __future__ import annotations

import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image

URL = "http://127.0.0.1:8765"


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def expect_status(path, payload, status):
    try:
        call(path, payload)
    except urllib.error.HTTPError as exc:
        assert exc.code == status, (exc.code, status)
    else:
        raise AssertionError(f"Expected HTTP {status}: {path}")


def main():
    before = call('/api/state')
    original_racks = [{k: r[k] for k in ('x', 'y', 'yaw_deg')} for r in before['racks']]
    results = []
    try:
        paused = call('/api/control', {"running": False})
        time.sleep(.25)
        assert call('/api/state')['simulation_time_s'] == paused['simulation_time_s']
        step = call('/api/control', {"step": True})
        assert abs(step['simulation_time_s'] - paused['simulation_time_s'] - .05) < .001
        results.append('Pause and fixed 50 ms stepping')

        for camera in before['cameras']:
            current = call('/api/control', {"camera": camera})
            assert current['camera'] == camera
            deadline = time.monotonic()+10
            while True:
                current = call('/api/state')
                if current.get('frame_camera') == camera and current.get('frame_scene_version') == current['scene_version']:
                    break
                assert time.monotonic() < deadline, 'Camera worker did not publish the requested view'
                time.sleep(.1)
            with urllib.request.urlopen(URL + '/frame.jpg', timeout=10) as response:
                frame = response.read()
            with Image.open(io.BytesIO(frame)) as image:
                assert image.size == tuple(current['resolution'])
                assert sum(image.convert('RGB').getextrema()[i][1] - image.convert('RGB').getextrema()[i][0] for i in range(3)) > 100
        results.append('All five actual camera images')

        with urllib.request.urlopen(URL + '/stream', timeout=10) as response:
            sample = response.read(4096)
            assert b'--frame' in sample and b'Content-Type: image/jpeg' in sample and b'\xff\xd8' in sample
        results.append('Browser MJPEG stream contains live JPEG frames')

        current = call('/api/reset', {"seed": 12345, "rack_count": 4})
        assert current['rack_count'] == 4 and current['tube_count'] >= 16
        first = current['racks']
        assert call('/api/reset', {"seed": 12345, "rack_count": 4})['racks'] == first
        positions = [{k: r[k] for k in ('x', 'y', 'yaw_deg')} for r in first]
        positions[0]['yaw_deg'] = 35
        assert call('/api/racks', {"racks": positions})['racks'][0]['yaw_deg'] == 35
        results.append('Seed reproducibility, four racks, and custom rack rotation')

        changed = call('/api/control', {"arm": 'left', "targets_deg": [999, -35, 45, 12, 0, 30]})
        joint = changed['arms']['left']['joints'][0]
        assert abs(joint['target_deg'] - joint['max_deg']) < .01
        call('/api/control', {"preset": 'home', "running": True})
        time.sleep(.4)
        assert call('/api/state')['simulation_time_s'] > 0
        call('/api/control', {"preset": 'gentle'})
        time.sleep(.5)
        assert call('/api/state')['motion_test']
        call('/api/control', {"running": False, "preset": 'home'})
        results.append('Joint-limit clamping and motion preset')

        expect_status('/api/control', {"camera": 'missing'}, 422)
        expect_status('/api/control', {"arm": 'left', "targets_deg": [1, 2]}, 422)
        expect_status('/api/reset', {"seed": -1, "rack_count": 1}, 422)
        expect_status('/api/racks', {"racks": [{"x": 0, "y": .1, "yaw_deg": 0}] * 2}, 400)
        results.append('Invalid camera, controls, count/seed, and overlapping racks rejected')
    finally:
        if before.get('scenario') == 'dinner':
            call('/api/reset', {"scenario": 'dinner', "seed": before['seed'], "dinner_preset": before['dinner_preset'], "drawer_open": before['drawer_open']})
        else:
            call('/api/reset', {"seed": before['seed'], "rack_count": before['rack_count'], "practice": before.get('practice', False), "transfer_side": before.get('transfer_side')})
            call('/api/racks', {"racks": original_racks})
        for arm in ('left', 'right'):
            call('/api/control', {"arm": arm, "targets_deg": [j['target_deg'] for j in before['arms'][arm]['joints']]})
        call('/api/control', {"running": before['running'], "camera": before['camera'], "shadows": before['shadows']})
    output = {"passed": results, "note": "Physics reset; original seed, rack poses, camera, running state and joint targets restored."}
    Path('.run').mkdir(exist_ok=True)
    Path('.run/live-check.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

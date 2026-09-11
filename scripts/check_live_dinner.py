"""Live dinner API checks. Resets the scene; restores its setup afterward."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_live_lab import call, expect_status


def main():
    before = call('/api/state')
    passed = []
    try:
        first = call('/api/reset', {"scenario": "dinner", "seed": 42})
        assert first['scenario'] == 'dinner' and first['object_count'] == 7
        assert first['stable_object_count'] == 7 and not first['task']['active']
        assert not first['drawer']['actuated']
        same = call('/api/reset', {"scenario": "dinner", "seed": 42})
        assert [o['initial_position_m'] for o in first['objects']] == [o['initial_position_m'] for o in same['objects']]
        other = call('/api/reset', {"scenario": "dinner", "seed": 43})
        assert [o['initial_position_m'] for o in first['objects']] != [o['initial_position_m'] for o in other['objects']]
        passed.append('Seeded dinner scene, seven physical objects and no active autonomous task')
        opened = call('/api/reset', {"scenario": "dinner", "drawer_open": True})
        assert abs(opened['drawer']['open_m']-.095) < .001
        assert all(o['position_m'][1] < .20 for o in opened['objects'] if o['id'] in ('fork', 'spoon'))
        passed.append('Open inspection preset exposes loose cutlery')
        reference = call('/api/reset', {"scenario": "dinner", "dinner_preset": "reference"})
        assert reference['dinner_preset'] == 'reference' and reference['task']['status'] == 'idle'
        for target in reference['targets']:
            obj = next(o for o in reference['objects'] if o['id'] == target['object_id'])
            assert max(abs(obj['position_m'][i]-target['position_m'][i]) for i in (0, 1)) < .002
        passed.append('Target example initializes placement without claiming task success')
        expect_status('/api/task', {"action": "start"}, 400)
        expect_status('/api/reset', {"scenario": "dinner", "practice": True}, 400)
        expect_status('/api/reset', {"scenario": "dinner", "dinner_preset": "reference", "drawer_open": True}, 400)
        expect_status('/api/reset', {"scenario": "dinner", "dinner_preset": "missing"}, 422)
        expect_status('/api/racks', {"racks": [{"x": -.18, "y": .1, "yaw_deg": 0}, {"x": .18, "y": .1, "yaw_deg": 0}]}, 400)
        assert call('/api/state')['scenario'] == 'dinner'
        passed.append('Unsupported dinner goals and incompatible rack controls fail clearly without losing the scene')
        chemistry = call('/api/reset', {"scenario": "chemistry", "transfer_side": "left"})
        assert chemistry['tube_count'] == 5 and chemistry['transfer_side'] == 'left'
        assert 'objects' not in chemistry
        assert call('/api/reset', {"scenario": "dinner"})['object_count'] == 7
        passed.append('Dinner and chemistry scenes switch without stale object state')
    finally:
        if before.get('scenario') == 'dinner':
            call('/api/reset', {"scenario": "dinner", "seed": before['seed'], "dinner_preset": before['dinner_preset'], "drawer_open": before['drawer_open']})
        else:
            call('/api/reset', {"seed": before['seed'], "rack_count": before['rack_count'], "practice": before.get('practice', False), "transfer_side": before.get('transfer_side')})
            call('/api/racks', {"racks": [{k: r[k] for k in ('x', 'y', 'yaw_deg')} for r in before['racks']]})
        for side in ('left', 'right'):
            call('/api/control', {"arm": side, "targets_deg": [j['target_deg'] for j in before['arms'][side]['joints']]})
        call('/api/control', {"running": before['running'], "camera": before['camera'], "shadows": before['shadows']})
    result = {"passed": passed, "scope": "Scene/API checks only. Original setup restored; physical trajectory reset."}
    Path('.run/dinner-live-check.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

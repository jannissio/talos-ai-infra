"""Check unchanged glass final independently of the conservative old mug sweep."""
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import prepare, freeze, put

out = ROOT / '.run/glass-clearance-final-geometry-v1'; log = prepare(out)
with log.open('x') as stream, contextlib.redirect_stdout(stream):
    source = ROOT / '.run/glass-clearance-order-geometry-v1'
    teacher, whole = freeze(out, source)
    put(out / 'source/scripts/check_glass_clearance_final.py', Path(__file__).read_bytes())
    put(out / 'protocol.json', dict(scope=__doc__, physical_steps=0,
        note='The former sweep filter alone does not classify the unchanged final setting unsolved; test its common source/transfer plan directly.'))
    model = mujoco.MjModel.from_xml_string((source / 'scene.xml').read_text())
    data = mujoco.MjData(model); layout = json.loads((source / 'layout.json').read_text())
    with np.load(source / 'actual-source-state.npz') as archive:
        data.qpos[:] = archive['qpos']; data.qvel[:] = archive['qvel']; data.ctrl[:] = archive['ctrl']
    mujoco.mj_forward(model, data)
    excluded = json.loads((source / 'prior-failed-grasp-exclusions.json').read_text())['excluded']
    from simulation_lab.scene import HOME
    task = teacher.TableTeacher(model, data, layout)
    task.start(object_id='glass', target=teacher.destination(layout, 'glass'), excluded_grasps=excluded)
    task.update(np.asarray(HOME * 2))
    report = dict(planned=task.active, message=task.message, search=task.search_log,
                  target=teacher.destination(layout, 'glass').tolist(), physical_steps=0)
    if task.active:report.update(candidate=task.chosen.name, side=task.side, roll=task.chosen_roll)
    put(out / 'report.json', report)
    print({k: v for k, v in report.items() if k != 'search'})

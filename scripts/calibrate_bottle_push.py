"""Exposed calibration of a disclosed base-height horizontal bottle push."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.scene import HOME, build_scene
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, 4*1024**2); args.output.mkdir(parents=True)
    seed = 2026095102; rng = np.random.default_rng(seed)
    x, y, yaw = rng.uniform(-.14, .16), rng.uniform(-.18, -.06), rng.uniform(-.6, .6)
    xml, _ = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
    rows = []
    if any(not .1 <= amplitude <= 2. for amplitude in args.amplitudes):
        raise ValueError('Choose a calibration force between 0.1 and 2 N.')
    for amplitude in args.amplitudes:
        require_space(args.output, 1024**2)
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
        address = model.joint('bottle_free').qposadr[0]; body = model.body('bottle').id
        data.qpos[address:address+7] = [x, y, .761, math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        mujoco.mj_forward(model, data)
        for _ in range(300):
            mujoco.mj_step(model, data)
        origin = data.body('bottle').xpos.copy(); max_tilt = 0.; peak_translation = 0.
        force = np.array([amplitude, 0., 0.])
        for tick in range(240):
            data.xfrc_applied[:] = 0.
            if tick < 20:
                point = data.body('bottle').xpos+np.array([0., 0., .012])
                data.xfrc_applied[body, :3] = force
                data.xfrc_applied[body, 3:] = np.cross(point-data.xipos[body], force)
            mujoco.mj_step(model, data)
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(data.body('bottle').xmat[8], -1, 1)))))
            peak_translation = max(peak_translation, float(np.linalg.norm(data.body('bottle').xpos[:2]-origin[:2])))
        translation = (data.body('bottle').xpos-origin)*1000
        rows.append({'amplitude_n': amplitude, 'duration_s': .1, 'application_height_m': .012,
                     'translation_mm': translation.tolist(), 'peak_translation_mm': peak_translation*1000,
                     'peak_tilt_deg': max_tilt, 'upright': bool(data.body('bottle').xmat[8] > .98)})
    result = {'seed': seed, 'scope': 'Exposed disturbance calibration with parked arms; not a controller success.',
              'force_direction': '+world X', 'hidden_forces': 0, 'disclosed_force': True, 'state_teleports_during_push': 0, 'rows': rows}
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--amplitudes', nargs='+', type=float, default=[.6, .8, 1., 1.2, 1.5, 2.])
    run(p.parse_args())

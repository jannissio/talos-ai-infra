"""Independently verify saved mug visibility images, masks and geometry; package all."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
from scripts.probe_mug_visual_visibility import (fixed_camera, read, rgb_component, sha,
                                                space, verify_sources, write_json)


def run(args):
    protocol_path = ROOT/'docs/robotics/experiments/mug-visual-visibility-v1.json'
    p = read(protocol_path)
    raw, package = ROOT/p['raw_root'], ROOT/p['evidence_package']
    report = read(raw/'report.json')
    if sha(protocol_path) != report['protocol_sha256'] or sha(raw/'masks.npz') != report['masks_sha256']:
        raise ValueError('Diagnostic inputs changed.')
    if sha(ROOT/'scripts/probe_mug_visual_visibility.py') != report['script_sha256']:
        raise ValueError('The frozen diagnostic changed.')
    for name, digest in report['input_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A recorded input changed: '+name)
    verify_sources(p)
    if len(report['states']) != 144 or len(report['views']) != 1296:
        raise ValueError('The complete declared diagnostic is required.')
    if package.exists() and not args.verify_only:
        raise FileExistsError('Preserve the earlier package.')
    if not args.verify_only:
        space(p, package, sum(path.stat().st_size for path in raw.iterdir() if path.is_file())+4*1024**2)
    width, height = p['image_size']
    with np.load(raw/'masks.npz', allow_pickle=False) as archive:
        predicted = np.unpackbits(archive['predicted'], axis=1).reshape(-1, height, width).astype(bool)
        truth = np.unpackbits(archive['scoring_only'], axis=1).reshape(-1, height, width).astype(bool)
    view_lookup = {(r['state_index'], r['configuration'], r['slot']): (i, r) for i, r in enumerate(report['views'])}
    recomputed = {name: [] for name in p['camera_configurations']}
    discrepancies, checked_images, checked_masks = [], 0, 0
    current, renderer = None, None
    try:
        for state_id, state in enumerate(report['states']):
            folder = ROOT/state['trace_root']/state['trace']
            if current != folder:
                if renderer is not None:
                    renderer.close()
                current = folder
                model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
                model.vis.quality.offsamples = 0
                data = mujoco.MjData(model)
                with np.load(folder/'states.npz', allow_pickle=False) as archive:
                    frames = {name: archive[name] for name in archive.files}
                renderer = mujoco.Renderer(model, width=width, height=height)
                option = mujoco.MjvOption()
                option.geomgroup[3:] = 0
                ids = np.flatnonzero(model.geom_bodyid == model.body('mug').id)
            index = state['frame_index']
            data.qpos[:], data.qvel[:], data.ctrl[:] = frames['qpos'][index], frames['qvel'][index], frames['targets'][index]
            data.time = float(frames['time'][index])
            mujoco.mj_forward(model, data)
            path = raw/state['mosaic']
            if sha(path) != state['mosaic_sha256']:
                raise ValueError('A retained image changed.')
            mosaic = np.asarray(Image.open(path).convert('RGB'))
            midpoint = data.body('mug').xpos+data.body('mug').xmat.reshape(3, 3)@np.asarray(p['centroid_diagnostic']['target_local_m'])
            for row, (name, cameras) in enumerate(p['camera_configurations'].items()):
                usable, areas, equations = [], [], []
                for slot, camera in enumerate(cameras):
                    mask_id, observation = view_lookup[state_id, name, slot]
                    pixels = mosaic[row*(height+22):row*(height+22)+height, slot*width:(slot+1)*width]
                    mask, inferred = rgb_component(pixels, observation['calibration']['projection'], p['rgb_method'])
                    if inferred != observation['rgb_observation'] or not np.array_equal(mask, predicted[mask_id]):
                        discrepancies.append([state_id, name, slot, 'image inference'])
                    renderer.update_scene(data, camera=fixed_camera(camera), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    if not np.array_equal(renderer.render(), pixels):
                        discrepancies.append([state_id, name, slot, 'rendered RGB'])
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    segmentation_mask = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 0], ids)
                    if not np.array_equal(segmentation_mask, truth[mask_id]):
                        discrepancies.append([state_id, name, slot, 'rendered truth mask'])
                    n, visible = int(mask.sum()), int(segmentation_mask.sum())
                    overlap = int((mask & segmentation_mask).sum())
                    precision, recall = overlap/max(n, 1), overlap/max(visible, 1)
                    g = p['visibility_gate']
                    passed = bool(n >= g['minimum_rgb_component_pixels'] and visible >= g['minimum_ground_truth_visible_pixels']
                                  and precision >= g['minimum_precision'] and recall >= g['minimum_recall'])
                    score = {'visible_pixels': visible, 'intersection_pixels': overlap, 'precision': precision, 'recall': recall, 'usable': passed}
                    if score != observation['scoring_only']:
                        discrepancies.append([state_id, name, slot, 'visibility score'])
                    if passed:
                        usable.append(slot)
                        areas.append(n)
                        u, v = inferred['centroid_px']
                        matrix = np.asarray(observation['calibration']['projection'])
                        equations.extend([u*matrix[2]-matrix[0], v*matrix[2]-matrix[1]])
                    checked_images += 1
                    checked_masks += 1
                accepted = len(usable) >= p['visibility_gate']['minimum_usable_views_per_state']
                expected = state['configurations'][name]
                second = sorted(areas, reverse=True)[1] if accepted else 0
                error = None
                if accepted:
                    _, _, vectors = np.linalg.svd(np.asarray(equations))
                    estimated = vectors[-1, :3]/vectors[-1, 3]
                    error = float(np.linalg.norm(estimated-midpoint)*1000)
                    if not np.allclose(midpoint, expected['scoring_only_midpoint_m'], atol=1e-12, rtol=0) or abs(error-expected['scoring_only_centroid_error_mm']) > 1e-9:
                        discrepancies.append([state_id, name, 'centroid geometry'])
                if (len(usable), accepted, second) != (expected['usable_views'], expected['visibility_gate_passed'], expected['second_largest_usable_component_pixels']):
                    discrepancies.append([state_id, name, 'state gate'])
                recomputed[name].append((accepted, second, error, state['phase']))
    finally:
        if renderer is not None:
            renderer.close()
    summaries = {}
    for name, values in recomputed.items():
        accepted = [r for r in values if r[0]]
        errors = [r[2] for r in accepted]
        summaries[name] = {'states': len(values), 'accepted': len(accepted), 'coverage_fraction': len(accepted)/len(values),
            'median_second_largest_usable_component_pixels': float(np.median([r[1] for r in accepted])) if accepted else 0,
            'gate_passed': len(accepted)/len(values) >= p['visibility_gate']['minimum_state_coverage_fraction'],
            'scoring_only_centroid_error_p95_mm': float(np.quantile(errors, .95)) if errors else None,
            'scoring_only_centroid_error_max_mm': max(errors) if errors else None,
            'phase_coverage': {phase: sum(r[0] for r in values if r[3] == phase) for phase in p['sample_labels']}}
    eligible = [n for n, r in summaries.items() if r['gate_passed']]
    selected = min(eligible, key=lambda n: (-summaries[n]['accepted'], -summaries[n]['median_second_largest_usable_component_pixels'], list(summaries).index(n))) if eligible else None
    if summaries != report['summaries'] or selected != report['selected_configuration']:
        discrepancies.append(['aggregate or selection'])
    audit = {'protocol_sha256': report['protocol_sha256'], 'source_sha256': sha(Path(__file__)),
        'independently_rendered_rgb_views': checked_images, 'independently_rendered_scoring_masks': checked_masks,
        'source_states': len(report['states']), 'discrepancies': discrepancies, 'selected_configuration': selected,
        'summaries': summaries, 'passed': not discrepancies}
    if discrepancies:
        raise ValueError('Independent audit failed: '+json.dumps(discrepancies[:10]))
    if args.verify_only:
        manifest = read(package/'manifest.json')
        for name, digest in manifest['files'].items():
            if sha(package/name) != digest:
                raise ValueError('Packaged artifact changed: '+name)
        print(json.dumps({'verified_files': len(manifest['files']), **audit}))
        return
    package.mkdir(parents=True)
    for source in [*sorted(raw.iterdir()), Path(__file__)]:
        if source.is_file():
            space(p, package/source.name, source.stat().st_size+1024)
            with (package/source.name).open('xb') as stream:
                stream.write(source.read_bytes())
    space(p, package/'audit.json', 1024**2)
    write_json(package/'audit.json', audit)
    files = {path.relative_to(package).as_posix(): sha(path) for path in package.rglob('*') if path.is_file()}
    write_json(package/'manifest.json', {'schema': p['schema'], 'files': files,
                                       'bytes': sum((package/name).stat().st_size for name in files)})
    space(p, package, 0)
    print(json.dumps({'packaged_files': len(files), 'packaged_bytes': sum((package/n).stat().st_size for n in files),
                      'independent_rgb_views': checked_images, 'discrepancies': discrepancies,
                      'selected_configuration': selected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args())

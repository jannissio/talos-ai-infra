"""Measure the adapted spatial policy on a real joint-scene observation.

This executes a single numerical optimizer step per head solely as a memory
and gradient smoke check. It creates no trained policy or robotics result.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party')]
import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from simulation_lab.random_dinner import draw
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_observation import TableObserver
from talos_cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from talos_cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from talos_cliport.utils.utils import preprocess


def put(path, value):
    with path.open('x', encoding='utf-8') as stream:json.dump(value, stream, indent=2)


def run(folder, checkpoint, rotations):
    if folder.exists():raise FileExistsError('Preserve this attempted runtime check.')
    preflight = require_space(folder, 128*1024**2); folder.mkdir(parents=True)
    report = {'scope': 'Architecture compatibility/memory smoke only; no trained robotics policy.',
              'preflight': preflight, 'rotations': rotations, 'heads': [], 'passed': False,
              'versions': {n: importlib.metadata.version(n) for n in
                ('torch', 'torchvision', 'mujoco', 'numpy', 'kornia', 'ftfy', 'regex')},
              'sources_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in list((ROOT/'third_party/talos_cliport').rglob('*.py'))+
                  [ROOT/'simulation_lab/table_observation.py', Path(__file__)]}}
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4); torch.manual_seed(2026115001)
        report['gpu'] = torch.cuda.get_device_name(0)
        report['total_vram_bytes'] = torch.cuda.get_device_properties(0).total_memory
        xml, _ = build_scene(seed=2026114001, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml); data, recipe = draw(model, 2026114001)
        assert recipe['generated']
        for _ in range(600):mujoco.mj_step(model, data)
        observer = TableObserver(model)
        started = time.perf_counter()
        try:observation = observer.observe(data)
        finally:observer.close()
        report['observation_wall_s'] = time.perf_counter()-started
        report['observed_cells'] = int(observation['observed'].sum())
        img = observation['policy_image']; report['image_shape'] = list(img.shape)
        require_space(folder, img.nbytes+2*1024**2)
        with (folder/'observation.npz').open('xb') as stream:
            np.savez_compressed(stream, policy_image=img, observed=observation['observed'])
        with (folder/'heightmap.png').open('xb') as stream:Image.fromarray(observation['rgb']).save(stream, format='PNG')
        put(folder/'calibration.json', {'grid': observation['grid'],
          'cameras': [{'name': v['name'], 'calibration': v['calibration']} for v in observation['views']]})
        cfg = {'clip_checkpoint_path': str(checkpoint), 'train': {'batchnorm': False,
          'lang_fusion_type': 'mult', 'attn_stream_fusion_type': 'add', 'trans_stream_fusion_type': 'conv'}}
        device = torch.device('cuda'); streams = ('plain_resnet_lat', 'clip_lingunet_lat')
        pick = TwoStreamAttentionLangFusionLat(streams, img.shape, 1, preprocess, cfg, device).to(device)
        place = TwoStreamTransportLangFusionLat(streams, img.shape, rotations, 64, preprocess, cfg, device).to(device)
        heads = nn.ModuleDict({'pick': pick, 'place': place}).train()
        trainable = list(p for p in heads.parameters() if p.requires_grad)
        report['trainable_parameters'] = sum(p.numel() for p in trainable)
        report['frozen_parameters'] = sum(p.numel() for p in heads.parameters() if not p.requires_grad)
        optimizers = {name: torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-4)
                      for name, head in heads.items()}
        scaler = torch.amp.GradScaler('cuda')
        torch.cuda.reset_peak_memory_stats()
        for name, head in heads.items():
            optimizers[name].zero_grad(set_to_none=True)
            before = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = head(img, 'place the bottle', softmax=False) if name == 'pick' else head(img, (120, 160), 'place the bottle', softmax=False)
                # Arbitrary test label; deliberately not saved or claimed as a demonstration.
                loss = F.cross_entropy(logits.reshape(1, -1).float(), torch.tensor([1234], device=device))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizers[name])
            gradients = [p.grad for p in head.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
            scaler.step(optimizers[name]); scaler.update(); torch.cuda.synchronize()
            report['heads'].append({'name': name, 'logit_shape': list(logits.shape),
                'loss': float(loss.detach()), 'wall_s': time.perf_counter()-before,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'finite_gradients': True})
            print(report['heads'][-1], flush=True)
            del logits, loss, gradients
            optimizers[name].zero_grad(set_to_none=True)
        report['passed'] = True
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if torch.cuda.is_available():
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        put(folder/'report.json', report)
        print({'passed': report['passed'], 'scope': report['scope']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--rotations', type=int, default=36)
    args = parser.parse_args(); run(args.output, args.checkpoint, args.rotations)

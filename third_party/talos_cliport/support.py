"""Original Talos compatibility helpers (MIT); no upstream GPL code included."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

RN50_SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
_BACKBONES = {}


class UpsampleMerge(nn.Module):
    """Bilinear feature expansion, skip concatenation and two local convolutions.

    Authored for the documented CLIPort decoder interface without reading or
    copying its GPL U-Net helper. This adaptation requires its own validation.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if not bilinear:
            raise ValueError('This independently authored helper supports bilinear expansion only.')
        middle = in_channels//2
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, middle, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
            nn.Conv2d(middle, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, coarse, skip):
        up = F.interpolate(coarse, scale_factor=2, mode='bilinear', align_corners=True)
        dh, dw = skip.shape[-2]-up.shape[-2], skip.shape[-1]-up.shape[-1]
        up = F.pad(up, (dw//2, dw-dw//2, dh//2, dh-dh//2))
        return self.mix(torch.cat((skip, up), dim=1))


class FrozenBackbone(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def visual(self):
        return self.core.visual

    def encode_text_with_embeddings(self, tokens):
        return self.core.encode_text_with_embeddings(tokens)


def shared_backbone(path, device):
    """One frozen RN50 per explicit file/device; no network or implicit cache."""
    path = Path(path).resolve(); key = (str(path), str(device))
    if key not in _BACKBONES:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != RN50_SHA256:
            raise ValueError('RN50 checkpoint does not match the published SHA256.')
        from .models.core.clip import build_model
        # TorchScript is read only from the exact official, hash-verified asset.
        original = torch.jit.load(str(path), map_location='cpu').eval()
        core = build_model(original.state_dict())
        del original
        _BACKBONES[key] = FrozenBackbone(core).to(device)
    return _BACKBONES[key]

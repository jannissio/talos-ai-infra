"""Compatibility safeguards for the selected pretrained representation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party'))
import torch
from torch import nn

from talos_cliport.support import FrozenBackbone, UpsampleMerge
from talos_cliport.models.core.clip import _download


class SpatialReferenceTests(unittest.TestCase):
    def test_frozen_encoder_preserves_batchnorm_during_decoder_training(self):
        core = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 4, 1))
        model = nn.ModuleDict({'encoder': FrozenBackbone(core), 'decoder': nn.Conv2d(4, 1, 1)})
        before = core[0].running_mean.clone()
        model.train()
        output = model['decoder'](model['encoder'].core(torch.ones(2, 3, 4, 4)))
        output.sum().backward()
        self.assertFalse(model['encoder'].training)
        self.assertFalse(core.training)
        self.assertTrue(torch.equal(before, core[0].running_mean))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in core.parameters()))
        self.assertIsNotNone(model['decoder'].weight.grad)

    def test_original_skip_helper_supports_odd_shapes_and_backpropagation(self):
        block = UpsampleMerge(24, 6)
        coarse = torch.randn(2, 12, 4, 5, requires_grad=True)
        skip = torch.randn(2, 12, 9, 11, requires_grad=True)
        result = block(coarse, skip)
        self.assertEqual(tuple(result.shape), (2, 6, 9, 11))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(coarse.grad).all())
        self.assertTrue(torch.isfinite(skip.grad).all())

    def test_implicit_upstream_download_is_disabled(self):
        with self.assertRaisesRegex(RuntimeError, 'implicit downloads are disabled'):
            _download('unused-test-url')


if __name__ == '__main__':
    unittest.main()

"""Training-free multi-guide extensions for the official CIFAR-10 ZeroPur."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def _kernel_size(sigma):
    return 2 * round(3 * sigma) + 1


def _blur(x, sigma):
    size = _kernel_size(sigma)
    return gaussian_blur(x, [size, size], [sigma, sigma])


class BiGuidePurifier(nn.Module):
    """Local graph guidance + high-frequency noise consensus + mutual fusion."""

    def __init__(self, classifier, normalizer):
        super().__init__()
        children = list(classifier.children())
        self.deep_features = nn.Sequential(
            normalizer,
            *children[:5],
            nn.AvgPool2d(4),
            nn.Flatten(1),
        )
        self.early_features = nn.Sequential(
            normalizer,
            *children[:2],
            nn.Flatten(1),
        )

    @staticmethod
    def _project(x, original, eps):
        x = torch.maximum(torch.minimum(x, original + eps), original - eps)
        return x.clamp(0, 1)

    @staticmethod
    def _graph_mask(batch, grid, ratio, graph_steps, device, generator):
        mask = (
            torch.rand(batch, 1, grid, grid, device=device, generator=generator)
            < ratio
        ).float()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
        ).view(1, 1, 3, 3)
        degree = F.conv2d(torch.ones_like(mask), kernel, padding=1).clamp_min(1)
        for _ in range(graph_steps):
            neighbor = F.conv2d(mask, kernel, padding=1) / degree
            mask = 0.5 * mask + 0.5 * neighbor
        return mask

    def _local_guides(
        self,
        x,
        count,
        grid,
        ratio,
        graph_steps,
        generator,
    ):
        global_blur = _blur(x, 1.8)
        strong_blur = _blur(x, 2.5)
        guides = []
        for _ in range(count):
            mask = self._graph_mask(
                len(x), grid, ratio, graph_steps, x.device, generator
            )
            mask = F.interpolate(
                mask, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            guides.append(global_blur * (1 - mask) + strong_blur * mask)
        return guides

    @staticmethod
    def _noise_guides(x, scale, generator):
        gaussian = torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        ) * scale
        uniform = (
            torch.rand(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            * 2
            - 1
        ) * scale
        rademacher = (
            torch.randint(
                0,
                2,
                x.shape,
                device=x.device,
                generator=generator,
            ).to(x.dtype)
            * 2
            - 1
        ) * scale
        return [(x + noise).clamp(0, 1) for noise in (gaussian, uniform, rademacher)]

    def _optimize(self, original, guide_fn, extractor, eps, steps, step_size):
        x = original.detach().clone()
        momentum = torch.zeros_like(x)
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            guides = guide_fn(x.detach())
            with torch.no_grad():
                targets = [extractor(guide).detach() for guide in guides]
            current = extractor(x)
            loss = torch.stack(
                [-F.cosine_similarity(current, target, dim=1).mean() for target in targets]
            ).mean()
            gradient = torch.autograd.grad(loss, x)[0]
            norm = gradient.abs().flatten(1).sum(1).view(-1, 1, 1, 1)
            gradient = gradient / norm.clamp_min(1e-12) + momentum
            momentum = gradient.detach()
            x = self._project(
                x.detach() - step_size * gradient.sign(), original, eps
            )
        return x.detach()

    def local_graph(
        self,
        inputs,
        generator,
        eps=10 / 255,
        steps=10,
        step_size=1 / 255,
        guides=3,
        grid=4,
        patch_ratio=0.25,
        graph_steps=2,
    ):
        return self._optimize(
            inputs,
            lambda x: self._local_guides(
                x, guides, grid, patch_ratio, graph_steps, generator
            ),
            self.deep_features,
            eps,
            steps,
            step_size,
        )

    def high_frequency_consensus(
        self,
        inputs,
        generator,
        eps=10 / 255,
        steps=10,
        step_size=1 / 255,
        noise_scale=4 / 255,
    ):
        return self._optimize(
            inputs,
            lambda x: self._noise_guides(x, noise_scale, generator),
            self.early_features,
            eps,
            steps,
            step_size,
        )

    def mutual_fusion(
        self,
        original,
        local,
        high_frequency,
        eps=10 / 255,
        steps=3,
        step_size=1 / 255,
        local_weight=0.75,
    ):
        with torch.no_grad():
            local_target = self.deep_features(local).detach()
            high_target = self.deep_features(high_frequency).detach()
        x = (local_weight * local + (1 - local_weight) * high_frequency).detach()
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            current = self.deep_features(x)
            loss = (
                -local_weight
                * F.cosine_similarity(current, local_target, dim=1).mean()
                -(1 - local_weight)
                * F.cosine_similarity(current, high_target, dim=1).mean()
            )
            gradient = torch.autograd.grad(loss, x)[0]
            x = self._project(
                x.detach() - step_size * gradient.sign(), original, eps
            )
        return x.detach()

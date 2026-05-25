"""Training utilities: focal loss, replay buffer, deterministic seeding."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 20260525) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        target_one = F.one_hot(target, num_classes=logits.shape[-1]).float()
        focal_weight = (1.0 - p) ** self.gamma
        loss = -(focal_weight * log_p * target_one).sum(dim=-1)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[target]
            loss = loss * alpha
        return loss.mean()


@dataclass
class ReplayBuffer:
    X: np.ndarray
    y: np.ndarray

    def stratified_sample(self, size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        classes, counts = np.unique(self.y, return_counts=True)
        per_class = max(1, size // max(len(classes), 1))
        picks: list[int] = []
        for c in classes:
            idx = np.where(self.y == c)[0]
            if len(idx) == 0:
                continue
            chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
            picks.extend(chosen.tolist())
        rng.shuffle(picks)
        picks = picks[:size]
        return self.X[picks], self.y[picks]

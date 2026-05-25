"""Multi-kernel 1D-CNN with attention pooling, trainable via CLI."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running both as `python -m src.model.cnn` and as a script via `python src/model/cnn.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline import (  # noqa: E402
    CHANNEL_NAMES,
    CLASSES,
    NUM_CHANNELS,
    NUM_CLASSES,
    WINDOW,
    build_training_set,
)
from src.model.training import FocalLoss, ReplayBuffer, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_V0_PATH = ARTIFACTS_DIR / "model_v0.pt"


class AttentionPool(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T] -> attn over T
        z = x.transpose(1, 2)  # [B, T, C]
        scores = self.score(z).squeeze(-1)  # [B, T]
        attn = torch.softmax(scores, dim=-1)  # [B, T]
        ctx = (z * attn.unsqueeze(-1)).sum(dim=1)  # [B, C]
        return ctx, attn


class MultiKernelCNN(nn.Module):
    def __init__(self, in_channels: int = NUM_CHANNELS, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch9 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),
            nn.ReLU(),
        )
        self.attention = AttentionPool(64)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C] -> transpose to [B, C, T]
        x = x.transpose(1, 2)
        h = torch.cat([self.branch3(x), self.branch5(x), self.branch9(x)], dim=1)
        h = self.fuse(h)
        ctx, attn = self.attention(h)
        logits = self.head(self.dropout(ctx))
        return logits, attn


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    return torch.tensor(inv / inv.sum() * NUM_CLASSES, dtype=torch.float32)


def _train_loop(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int = 32,
    class_alpha: torch.Tensor | None = None,
    progress: callable | None = None,
) -> dict[str, list[float]]:
    device = torch.device("cpu")
    model.to(device).train()
    loss_fn = FocalLoss(gamma=1.5, alpha=class_alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    history = {"loss": [], "acc": []}
    n = len(X_t)
    for epoch in range(epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            xb = X_t[sel].to(device)
            yb = y_t[sel].to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach()) * len(sel)
            correct += int((logits.argmax(-1) == yb).sum())
        avg_loss = epoch_loss / n
        acc = correct / n
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        if progress is not None:
            progress(epoch + 1, epochs, avg_loss, acc)
    return history


def train_initial(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 20260525,
    progress: callable | None = None,
) -> tuple[MultiKernelCNN, dict[str, list[float]]]:
    set_seed(seed)
    model = MultiKernelCNN()
    alpha = _class_weights(y)
    history = _train_loop(model, X, y, epochs=epochs, lr=lr, class_alpha=alpha, progress=progress)
    return model, history


def fine_tune(
    model: MultiKernelCNN,
    new_X: np.ndarray,
    new_y: np.ndarray,
    replay: ReplayBuffer | None,
    epochs: int = 6,
    lr: float = 3e-4,
    replay_size: int = 64,
    progress: callable | None = None,
) -> tuple[MultiKernelCNN, dict[str, list[float]]]:
    new_X = np.asarray(new_X, dtype=np.float32)
    new_y = np.asarray(new_y, dtype=np.int64)
    if replay is not None:
        rx, ry = replay.stratified_sample(replay_size, seed=int(time.time()) % 10000)
        X = np.concatenate([new_X, rx], axis=0)
        y = np.concatenate([new_y, ry], axis=0)
    else:
        X = new_X
        y = new_y
    alpha = _class_weights(y)
    history = _train_loop(model, X, y, epochs=epochs, lr=lr, class_alpha=alpha, progress=progress)
    return model, history


def predict_with_attn(model: MultiKernelCNN, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        logits, attn = model(xb)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs, attn.cpu().numpy()


def channel_occlusion(model: MultiKernelCNN, x: np.ndarray, target_class: int) -> dict[str, float]:
    """Mask each channel individually, measure target-class probability drop."""
    model.eval()
    with torch.no_grad():
        baseline = predict_with_attn(model, x[None, ...])[0][0, target_class]
        results: dict[str, float] = {}
        means = x.mean(axis=0)
        for ci, name in enumerate(CHANNEL_NAMES):
            occluded = x.copy()
            occluded[:, ci] = means[ci]
            prob = predict_with_attn(model, occluded[None, ...])[0][0, target_class]
            results[name] = float(baseline - prob)
        return results


def snapshot(model: MultiKernelCNN) -> dict[str, torch.Tensor]:
    return copy.deepcopy(model.state_dict())


def load_snapshot(state: dict[str, torch.Tensor]) -> MultiKernelCNN:
    model = MultiKernelCNN()
    model.load_state_dict(state)
    model.eval()
    return model


def _accuracy(model: MultiKernelCNN, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = predict_with_attn(model, X)
    pred = probs.argmax(-1)
    return float((pred == y).mean())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train initial model and save artifacts/model_v0.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if not args.train:
        parser.print_help()
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading training data ...")
    X, y, _ = build_training_set()
    print(f"  X shape = {X.shape}, classes = {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    def cb(epoch: int, total: int, loss: float, acc: float) -> None:
        print(f"  epoch {epoch:02d}/{total} loss={loss:.4f} acc={acc:.3f}")

    model, history = train_initial(X, y, epochs=args.epochs, lr=args.lr, progress=cb)
    final_acc = _accuracy(model, X, y)
    print(f"Final train accuracy = {final_acc:.3f}")

    torch.save({"state_dict": model.state_dict(), "history": history, "classes": list(CLASSES)}, MODEL_V0_PATH)
    print(f"Saved -> {MODEL_V0_PATH}")


if __name__ == "__main__":
    main()

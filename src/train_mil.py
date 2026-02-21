from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix

from datasets import H5BagDataset
from mil_models import AttentionMIL
from utils import set_seed


@dataclass(frozen=True)
class TrainConfig:
    bags_dir: str
    split_csv: str  # columns: patient_id, split in {train,val,test}, label
    out_dir: str
    feat_dim: int
    n_classes: int = 3
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    attn_dim: int = 256
    dropout: float = 0.25
    seed: int = 0


def macro_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
    return float(np.mean(aucs)) if len(aucs) else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> Dict:
    model.eval()
    ys, probs = [], []

    for feats, y, _pid in loader:
        feats = feats.squeeze(0).to(device)  # (N,D), batch_size=1
        logits, _w = model(feats)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        ys.append(int(y.item()))
        probs.append(p)

    y_true = np.asarray(ys, dtype=np.int64)
    y_prob = np.vstack(probs) if len(probs) else np.zeros((0, n_classes), dtype=np.float32)

    if len(y_true) == 0:
        return {"macro_auc_ovr": float("nan"), "macro_f1": float("nan"), "balanced_acc": float("nan"), "confusion": []}

    y_pred = y_prob.argmax(axis=1)
    return {
        "macro_auc_ovr": macro_ovr_auc(y_true, y_prob, n_classes),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.split_csv)
    train_ids = df.loc[df.split == "train", "patient_id"].astype(str).tolist()
    val_ids = df.loc[df.split == "val", "patient_id"].astype(str).tolist()
    test_ids = df.loc[df.split == "test", "patient_id"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = H5BagDataset(cfg.bags_dir, train_ids)
    val_ds = H5BagDataset(cfg.bags_dir, val_ids)
    test_ds = H5BagDataset(cfg.bags_dir, test_ids)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = AttentionMIL(
        in_dim=cfg.feat_dim,
        attn_dim=cfg.attn_dim,
        n_classes=cfg.n_classes,
        dropout=cfg.dropout
    ).to(device)

    # class weights for imbalance (HER2-high often rarer)
    train_labels = df.loc[df.split == "train", "label"].astype(int).to_numpy()
    counts = np.bincount(train_labels, minlength=cfg.n_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for feats, y, _pid in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            feats = feats.squeeze(0).to(device)  # (N,D)
            y = y.to(device).view(())            # scalar

            logits, _w = model(feats)
            loss = ce(logits.unsqueeze(0), y.unsqueeze(0))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device, cfg.n_classes)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val={val_metrics}")

        score = val_metrics["macro_auc_ovr"]
        if np.isfinite(score) and score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

    # Test with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, cfg.n_classes)

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_macro_auc": best_val, "test": test_metrics}, f, indent=2)

    print("TEST:", test_metrics)


if __name__ == "__main__":
    # If you used resnet50 encoder, feat_dim is typically 2048.
    cfg = TrainConfig(
        bags_dir="bags_resnet50_mpp0p5",
        split_csv="configs/tcga_example.csv",
        out_dir="runs/abmil_resnet50_mpp0p5",
        feat_dim=2048,
        epochs=20,
        lr=1e-4,
        seed=0,
    )
    main(cfg)

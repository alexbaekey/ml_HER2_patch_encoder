from __future__ import annotations

import os
import math
import json
import random
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import timm
import openslide

# ctranspath
from timm.layers import to_2tuple

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("HEROHE")
TRAIN_XLSX = ROOT / "Training (ground truth).xlsx"
TEST_XLSX = ROOT / "Test (ground truth).xlsx"

OUT_DIR = Path("herohe_ihc2_runs")

ENCODERS = [
    "ctranspath", # https://github.com/Xiyue-Wang/TransPath
    "resnet50",
    "vit_base_patch16_224",
    "convnext_base"
]

PATCH_SIZE = 256
TARGET_MAG = 20
MAX_PATCHES_PER_SLIDE = 50 # first run
#MAX_PATCHES_PER_SLIDE = 100 # second run
#MAX_PATCHES_PER_SLIDE = 500 # third run
#MAX_PATCHES_PER_SLIDE = 1000 # fourth run
#MAX_PATCHES_PER_SLIDE = 5000 # fourth run
#MAX_PATCHES_PER_SLIDE = 10000 # fifth run


BATCH_SIZE_PATCHES = 64
NUM_WORKERS = 4

VAL_FRAC = 0.2
SEED = 317

EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.25



# Initial test run
#ENCODERS = ["resnet50"]
#MAX_PATCHES_PER_SLIDE = 500
#EPOCHS = 5
#NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# UTIL
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clean_case_id(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


# ============================================================
# LABELS
# ============================================================

def load_ihc2_binary_labels(xlsx_path: Path) -> dict[str, int]:
    """
    Keep only IHC == 2.
    Label is Final Result (Ground truth):
      Negative -> 0
      Positive -> 1
    """
    df = pd.read_excel(xlsx_path)

    required = ["Case", "Immunohistochemistry", "Final Result (Ground truth)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {xlsx_path}. Found: {list(df.columns)}")

    df = df[df["Immunohistochemistry"] == 2].copy()

    labels = {}
    for _, row in df.iterrows():
        sid = clean_case_id(row["Case"])
        gt = str(row["Final Result (Ground truth)"]).strip().lower()

        if gt == "negative":
            y = 0
        elif gt == "positive":
            y = 1
        else:
            continue

        labels[sid] = y

    return labels


def build_slide_table(split_dir: Path, labels: dict[str, int], split_name: str) -> pd.DataFrame:
    rows = []
    for sid, y in labels.items():
        mrxs = split_dir / f"{sid}.mrxs"
        folder = split_dir / sid
        if mrxs.exists() and folder.exists():
            rows.append({
                "slide_id": sid,
                "split": split_name,
                "label": y,
                "mrxs_path": str(mrxs),
            })
    return pd.DataFrame(rows)


# ============================================================
# OPENSLIDE / PATCH SAMPLING
# ============================================================

def choose_level(slide: openslide.OpenSlide, target_mag: int = 20) -> int:
    """
    Best-effort selection of a pyramid level close to target magnification.
    If metadata is missing, default to level 0.
    """
    props = slide.properties

    # mpp-x is often available
    if "openslide.mpp-x" in props:
        try:
            mpp = float(props["openslide.mpp-x"])
            # crude mapping: 0.25 um/px ~ 40x, 0.5 um/px ~ 20x
            base_mag = 10.0 / mpp
            best_level = 0
            best_diff = float("inf")
            for lvl in range(slide.level_count):
                eff_mag = base_mag / float(slide.level_downsamples[lvl])
                diff = abs(eff_mag - target_mag)
                if diff < best_diff:
                    best_diff = diff
                    best_level = lvl
            return best_level
        except Exception:
            pass

    return 0


def make_tissue_mask(slide: openslide.OpenSlide, thumb_size: int = 2048):
    w, h = slide.dimensions
    scale = min(thumb_size / w, thumb_size / h, 1.0)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    arr = np.asarray(thumb).astype(np.float32) / 255.0

    # simple white-background removal
    mean = arr.mean(axis=2)
    tissue = mean < 0.85

    sx = w / tw
    sy = h / th
    return tissue, sx, sy


def sample_coords(slide: openslide.OpenSlide, patch_size: int, max_patches: int, seed: int):
    tissue, sx, sy = make_tissue_mask(slide)

    ys, xs = np.where(tissue)
    if len(xs) == 0:
        return []

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(max_patches, len(xs)), replace=False)

    coords = []
    for i in idx:
        xt = int(xs[i])
        yt = int(ys[i])

        x0 = int(xt * sx - patch_size // 2)
        y0 = int(yt * sy - patch_size // 2)

        x0 = max(0, x0)
        y0 = max(0, y0)
        coords.append((x0, y0))

    return coords


# ============================================================
# ENCODERS
# ============================================================

class ConvStem(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        **kwargs
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim = 3
        output_dim = embed_dim // 8

        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3,
                                  stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2

        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))

        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                 # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)       # [B, H, W, C]
        x = self.norm(x)
        return x


class FrozenEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

        if name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet50(weights=weights)
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.out_dim = 2048
            self.transform = weights.transforms()

        elif name == "vit_base_patch16_224":
            # Manual normalization values from ImageNet RGB images
            self.model = timm.create_model(name, pretrained=True, num_classes=0)
            self.out_dim = self.model.num_features
            self.transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])

        elif name == "convnext_base":
            # Manual normalization values from ImageNet RGB images
            self.model = timm.create_model(name, pretrained=True, num_classes=0)
            self.out_dim = self.model.num_features
            self.transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])

        elif name == "ctranspath":
            CKPT_PATH = "checkpoints/ctranspath.pth"

            self.model = timm.create_model(
                "swin_tiny_patch4_window7_224",
                embed_layer=ConvStem,
                pretrained=False,
                num_classes=0,
            )

            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            new_state = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "").replace("encoder.", "")

                # remove old timm buffers
                if "relative_position_index" in k or "attn_mask" in k:
                    continue

                # old timm placed downsample at layer i;
                # new timm places it at layer i+1
                if k.startswith("layers.0.downsample."):
                    k = k.replace("layers.0.downsample.", "layers.1.downsample.")
                elif k.startswith("layers.1.downsample."):
                    k = k.replace("layers.1.downsample.", "layers.2.downsample.")
                elif k.startswith("layers.2.downsample."):
                    k = k.replace("layers.2.downsample.", "layers.3.downsample.")

                new_state[k] = v

            missing, unexpected = self.model.load_state_dict(new_state, strict=False)

            print("CTransPath missing keys:", len(missing))
            print("CTransPath unexpected keys:", len(unexpected))

            self.out_dim = self.model.num_features

            self.transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])


        else:
            raise ValueError(f"Unknown encoder: {name}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        y = self.model(x)
        if y.ndim == 4:
            y = y.flatten(1)
        return y


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, mrxs_path: str, coords: list[tuple[int, int]], level: int, patch_size: int, transform):
        self.mrxs_path = mrxs_path
        self.coords = coords
        self.level = level
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x0, y0 = self.coords[idx]
        slide = openslide.OpenSlide(self.mrxs_path)
        try:
            img = slide.read_region((x0, y0), self.level, (self.patch_size, self.patch_size)).convert("RGB")
        finally:
            slide.close()
        return self.transform(img), np.array([x0, y0], dtype=np.int64)


def extract_features_for_slide(row, encoder: FrozenEncoder, feature_dir: Path):
    slide_id = row["slide_id"]
    split = row["split"]
    label = int(row["label"])
    mrxs_path = row["mrxs_path"]

    out_path = feature_dir / split / f"{slide_id}.pt"
    ensure_dir(out_path.parent)

    if out_path.exists():
        return out_path

    slide = openslide.OpenSlide(mrxs_path)
    try:
        level = choose_level(slide, TARGET_MAG)
        coords = sample_coords(
            slide,
            patch_size=PATCH_SIZE,
            max_patches=MAX_PATCHES_PER_SLIDE,
            seed=SEED + int(slide_id),
        )
    finally:
        slide.close()

    if len(coords) == 0:
        payload = {
            "slide_id": slide_id,
            "label": label,
            "features": torch.empty((0, encoder.out_dim), dtype=torch.float32),
            "coords": torch.empty((0, 2), dtype=torch.int64),
        }
        torch.save(payload, out_path)
        return out_path

    ds = PatchDataset(
        mrxs_path=mrxs_path,
        coords=coords,
        level=level,
        patch_size=PATCH_SIZE,
        transform=encoder.transform,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE_PATCHES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    encoder = encoder.to(DEVICE)
    feats_all = []
    coords_all = []

    for xb, cb in tqdm(dl, leave=False, desc=f"extract {slide_id}"):
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            fb = encoder(xb).cpu()
        feats_all.append(fb)
        coords_all.append(cb)

    feats = torch.cat(feats_all, dim=0)
    coords = torch.cat(coords_all, dim=0)

    payload = {
        "slide_id": slide_id,
        "label": label,
        "features": feats,
        "coords": coords,
    }
    torch.save(payload, out_path)
    return out_path


# ============================================================
# MIL
# ============================================================

class FeatureBagDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #d = torch.load(self.paths[idx], map_location="cpu") #future warnings
        d = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        return d["features"].float(), int(d["label"]), d["slide_id"]


def collate_bags(batch):
    xs, ys, ids = zip(*batch)
    return list(xs), torch.tensor(ys, dtype=torch.long), list(ids)


class ABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, bag):
        h = torch.tanh(self.fc(bag))
        h = self.dropout(h)
        a = self.attn(h).squeeze(1)
        w = torch.softmax(a, dim=0)
        z = (w.unsqueeze(1) * h).sum(dim=0)
        logits = self.cls(z)
        return logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    for bags, ys, _ids in loader:
        for bag, y in zip(bags, ys.tolist()):
            bag = bag.to(DEVICE)
            logits = model(bag)
            prob = torch.softmax(logits, dim=0)[1].item()
            pred = int(torch.argmax(logits).item())

            y_true.append(y)
            y_pred.append(pred)
            y_prob.append(prob)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(set(y_true)) == 2:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)

    return metrics


def train_one_encoder(encoder_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(f"ENCODER: {encoder_name}")
    print("=" * 80)

    encoder = FrozenEncoder(encoder_name)

    feature_dir = OUT_DIR / "features" / encoder_name
    ensure_dir(feature_dir)

    # ------------------------------------------------
    # Extract train + test features
    # ------------------------------------------------
    print("\nExtracting train features...")
    train_paths = []
    for _, row in train_df.iterrows():
        p = extract_features_for_slide(row, encoder, feature_dir)
        train_paths.append(p)

    print("\nExtracting test features...")
    test_paths = []
    for _, row in test_df.iterrows():
        p = extract_features_for_slide(row, encoder, feature_dir)
        test_paths.append(p)

    # ------------------------------------------------
    # Train/val split from train set
    # ------------------------------------------------
    train_labels = train_df["label"].tolist()
    tr_paths, va_paths = train_test_split(
        train_paths,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=train_labels,
    )

    train_ds = FeatureBagDataset(tr_paths)
    val_ds = FeatureBagDataset(va_paths)
    test_ds = FeatureBagDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bags)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_bags)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_bags)

    model = ABMIL(
        in_dim=encoder.out_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout=DROPOUT,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_ckpt = OUT_DIR / "models" / f"{encoder_name}_best.pt"
    ensure_dir(best_ckpt.parent)

    # ------------------------------------------------
    # Train
    # ------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for bags, ys, _ids in tqdm(train_loader, desc=f"train {encoder_name} ep{epoch}", leave=False):
            bag = bags[0].to(DEVICE)
            y = ys[0].to(DEVICE)

            logits = model(bag)
            loss = F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={np.mean(losses):.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------
    # Final eval
    # ------------------------------------------------
    #model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE)) future warnings
    model.load_state_dict(
        torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    )

    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    print("\nBEST VALIDATION:")
    print(val_metrics)

    print("\nTEST:")
    print(test_metrics)

    result = {
        "encoder": encoder_name,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / f"results_{encoder_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    # load IHC==2 only, Positive/Negative labels
    train_labels = load_ihc2_binary_labels(TRAIN_XLSX)
    test_labels = load_ihc2_binary_labels(TEST_XLSX)

    print("Train class counts:", Counter(train_labels.values()))
    print("Test class counts:", Counter(test_labels.values()))

    train_df = build_slide_table(ROOT / "train", train_labels, "train")
    test_df = build_slide_table(ROOT / "test", test_labels, "test")

    print("Train slides found:", len(train_df))
    print("Test slides found:", len(test_df))

    all_results = []
    for enc in ENCODERS:
        res = train_one_encoder(enc, train_df, test_df)
        all_results.append(res)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for r in all_results:
        print(
            f"{r['encoder']:>22s} | "
            f"val_f1={r['val']['macro_f1']:.4f} | "
            f"test_acc={r['test']['accuracy']:.4f} | "
            f"test_f1={r['test']['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

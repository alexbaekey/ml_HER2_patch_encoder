from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import openslide


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("HEROHE")
RUNS_DIR = Path("herohe_ihc2_runs")
OUT_DIR = RUNS_DIR / "example_figures"

ENCODERS = [
    "resnet50",
    "vit_base_patch16_224",
    "convnext_base",
]

PATCH_SIZE = 256
TARGET_MAG = 20

NUM_CORRECT = 4
NUM_INCORRECT = 4
TOP_K_PATCHES = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================

class ABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, bag: torch.Tensor):
        h = torch.tanh(self.fc(bag))
        h = self.dropout(h)
        a = self.attn(h).squeeze(1)          # [N]
        w = torch.softmax(a, dim=0)          # [N]
        z = (w.unsqueeze(1) * h).sum(dim=0)  # [H]
        logits = self.cls(z)                 # [C]
        return logits, w


# ============================================================
# OPENSLIDE HELPERS
# ============================================================

def choose_level(slide: openslide.OpenSlide, target_mag: int = 20) -> int:
    props = slide.properties

    if "openslide.mpp-x" in props:
        try:
            mpp = float(props["openslide.mpp-x"])
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


def read_patch_from_coords(mrxs_path: Path, x0: int, y0: int, patch_size: int = 256, target_mag: int = 20) -> Image.Image:
    slide = openslide.OpenSlide(str(mrxs_path))
    try:
        level = choose_level(slide, target_mag=target_mag)
        img = slide.read_region((int(x0), int(y0)), level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()
    return img


# ============================================================
# INFERENCE
# ============================================================

def load_model(encoder_name: str) -> ABMIL:
    ckpt_path = RUNS_DIR / "models" / f"{encoder_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # infer dimensions from saved checkpoint if available
    if isinstance(ckpt, dict) and "fc.weight" not in ckpt:
        # if you saved whole state dict directly, this won't happen
        state_dict = ckpt
    else:
        state_dict = ckpt

    # infer in_dim from fc weight shape
    fc_w = state_dict["fc.weight"]
    hidden_dim, in_dim = fc_w.shape

    cls_w = state_dict["cls.weight"]
    num_classes = cls_w.shape[0]

    # infer dropout only from our known script; keep 0.25
    model = ABMIL(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=0.25)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_encoder_test_examples(encoder_name: str):
    """
    Returns list of dicts with:
      slide_id, label, pred, prob_pos, confidence, correct, top_coords
    """
    feature_dir = RUNS_DIR / "features" / encoder_name / "test"
    model = load_model(encoder_name)

    results = []

    for feat_path in sorted(feature_dir.glob("*.pt")):
        d = torch.load(feat_path, map_location="cpu", weights_only=False)

        bag = d["features"].float()
        coords = d["coords"]
        label = int(d["label"])
        slide_id = str(d["slide_id"])

        if bag.shape[0] == 0:
            continue

        logits, attn = model(bag.to(DEVICE))
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred = int(np.argmax(probs))
        prob_pos = float(probs[1]) if len(probs) > 1 else float(probs[0])

        # confidence = probability of predicted class
        confidence = float(np.max(probs))
        correct = (pred == label)

        # top attended patches
        attn_np = attn.detach().cpu().numpy()
        top_idx = np.argsort(-attn_np)[:TOP_K_PATCHES]
        top_coords = coords[top_idx].numpy()

        results.append({
            "slide_id": slide_id,
            "label": label,
            "pred": pred,
            "prob_pos": prob_pos,
            "confidence": confidence,
            "correct": correct,
            "top_coords": top_coords,
        })

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def label_str(y: int) -> str:
    return "Positive" if y == 1 else "Negative"


def add_border(img: Image.Image, color: str, width: int = 6) -> Image.Image:
    return ImageOps.expand(img, border=width, fill=color)


def choose_examples(results, num_correct=4, num_incorrect=4):
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    # show confident correct examples and confident incorrect examples
    correct = sorted(correct, key=lambda x: -x["confidence"])[:num_correct]
    incorrect = sorted(incorrect, key=lambda x: -x["confidence"])[:num_incorrect]

    return correct, incorrect


def make_encoder_figure(encoder_name: str, examples_correct, examples_incorrect):
    examples = examples_correct + examples_incorrect
    if len(examples) == 0:
        print(f"No examples found for {encoder_name}")
        return

    n_rows = len(examples)
    n_cols = TOP_K_PATCHES

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, ex in enumerate(examples):
        sid = ex["slide_id"]
        mrxs_path = ROOT / "test" / f"{sid}.mrxs"

        color = "green" if ex["correct"] else "red"
        status = "CORRECT" if ex["correct"] else "WRONG"

        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")

            x0, y0 = ex["top_coords"][c]
            img = read_patch_from_coords(mrxs_path, x0, y0, patch_size=PATCH_SIZE, target_mag=TARGET_MAG)
            img = add_border(img, color=color, width=4)

            ax.imshow(img)

            if c == 0:
                ax.set_ylabel(
                    f"{status}\nID={sid}\ntrue={label_str(ex['label'])}\npred={label_str(ex['pred'])}\nconf={ex['confidence']:.2f}",
                    rotation=0,
                    labelpad=80,
                    fontsize=9,
                    va="center"
                )

    fig.suptitle(f"{encoder_name}: top attended patches for correct and incorrect test examples", fontsize=14)
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.96])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{encoder_name}_examples.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for encoder_name in ENCODERS:
        print(f"\nProcessing {encoder_name}")
        results = run_encoder_test_examples(encoder_name)
        correct, incorrect = choose_examples(
            results,
            num_correct=NUM_CORRECT,
            num_incorrect=NUM_INCORRECT,
        )

        print(f"  correct examples shown:   {len(correct)}")
        print(f"  incorrect examples shown: {len(incorrect)}")

        make_encoder_figure(encoder_name, correct, incorrect)


if __name__ == "__main__":
    main()

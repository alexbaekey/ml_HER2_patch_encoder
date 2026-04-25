from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import openslide
from PIL import Image, ImageOps


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("HEROHE")
RUNS_DIR = Path("herohe_ihc2_runs")

OUT_DIR_GRID = RUNS_DIR / "example_figures"
OUT_DIR_SIMPLE = RUNS_DIR / "top_patch_figures"

ENCODERS = [
    "resnet50",
    "vit_base_patch16_224",
    "convnext_base",
    "ctranspath",
]

PATCH_SIZE = 256
TARGET_MAG = 20

NUM_CORRECT_GRID = 4
NUM_INCORRECT_GRID = 4
TOP_K_PATCHES = 6

NUM_CORRECT_SIMPLE = 5
NUM_INCORRECT_SIMPLE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================

class ABMIL(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=2, dropout=0.25):
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
        return logits, w


def load_model(encoder_name):
    ckpt_path = RUNS_DIR / "models" / f"{encoder_name}_best.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_dim, in_dim = state_dict["fc.weight"].shape
    num_classes = state_dict["cls.weight"].shape[0]

    model = ABMIL(in_dim, hidden_dim, num_classes, dropout=0.25)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# OPENSLIDE HELPERS
# ============================================================

def choose_level(slide, target_mag=20):
    if "openslide.mpp-x" in slide.properties:
        try:
            mpp = float(slide.properties["openslide.mpp-x"])
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


def read_patch_from_coords(mrxs_path, x0, y0, patch_size=256, target_mag=20):
    slide = openslide.OpenSlide(str(mrxs_path))
    try:
        level = choose_level(slide, target_mag=target_mag)
        img = slide.read_region((int(x0), int(y0)), level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()
    return img


def read_top_patch(mrxs_path, x0, y0, patch_size=256, target_mag=20):
    return read_patch_from_coords(mrxs_path, x0, y0, patch_size, target_mag)


def add_border(img, color, width=6):
    return ImageOps.expand(img, border=width, fill=color)


def add_border_simple(img, correct, width=5):
    color = "green" if correct else "red"
    return ImageOps.expand(img, border=width, fill=color)


def label_str(y):
    return "Positive" if y == 1 else "Negative"


# ============================================================
# COLLECT EXAMPLES
# ============================================================

@torch.no_grad()
def collect_examples_for_encoder(encoder_name):
    feature_dir = RUNS_DIR / "features" / encoder_name / "test"
    model = load_model(encoder_name)

    examples = []

    for feat_path in sorted(feature_dir.glob("*.pt")):
        d = torch.load(feat_path, map_location="cpu", weights_only=False)

        bag = d["features"].float()
        coords = d["coords"]
        slide_id = str(d["slide_id"])
        true_label = int(d["label"])

        if bag.shape[0] == 0:
            continue

        logits, attn = model(bag.to(DEVICE))
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))

        attn_np = attn.cpu().numpy()
        top_idx = np.argsort(-attn_np)[:TOP_K_PATCHES]
        top_coords = coords[top_idx].numpy()

        examples.append({
            "slide_id": slide_id,
            "label": true_label,
            "true_label": true_label,
            "pred": pred_label,
            "pred_label": pred_label,
            "prob_pos": float(probs[1]),
            "prob_positive": float(probs[1]),
            "confidence": confidence,
            "correct": pred_label == true_label,
            "num_patches": int(bag.shape[0]),
            "top_coords": top_coords,
            "top_x": int(top_coords[0][0]),
            "top_y": int(top_coords[0][1]),
        })

    return examples


def choose_examples(results, num_correct, num_incorrect):
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    correct = sorted(correct, key=lambda x: -x["confidence"])[:num_correct]
    incorrect = sorted(incorrect, key=lambda x: -x["confidence"])[:num_incorrect]

    return correct, incorrect


# ============================================================
# ORIGINAL GRID FIGURE STYLE
# ============================================================

def make_encoder_figure(encoder_name, examples_correct, examples_incorrect):
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

    OUT_DIR_GRID.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR_GRID / f"{encoder_name}_examples.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_path}")


# ============================================================
# ORIGINAL SIMPLE FIGURE STYLE
# ============================================================

def make_figure(encoder_name, examples, title, out_name):
    if len(examples) == 0:
        print(f"No examples for {encoder_name} / {title}")
        return

    nrows = len(examples)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 2.6 * nrows))

    if nrows == 1:
        axes = np.array([axes])

    for r, ex in enumerate(examples):
        ax_text = axes[r, 0]
        ax_img = axes[r, 1]

        ax_text.axis("off")
        ax_img.axis("off")

        text = (
            f"Slide ID: {ex['slide_id']}\n"
            f"True label: {label_str(ex['true_label'])}\n"
            f"Predicted: {label_str(ex['pred_label'])}\n"
            f"Confidence: {ex['confidence']:.3f}\n"
            f"P(Positive): {ex['prob_positive']:.3f}\n"
            f"Sampled patches: {ex['num_patches']}"
        )
        ax_text.text(0.0, 0.5, text, fontsize=11, va="center", ha="left")

        mrxs_path = ROOT / "test" / f"{ex['slide_id']}.mrxs"
        img = read_top_patch(
            mrxs_path,
            ex["top_x"],
            ex["top_y"],
            patch_size=PATCH_SIZE,
            target_mag=TARGET_MAG,
        )

        color = "green" if ex["correct"] else "red"
        img = add_border(img, color=color, width=5)
        ax_img.imshow(img)
        ax_img.set_title("Top attended patch", fontsize=10)

    fig.suptitle(f"{encoder_name} — {title}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR_SIMPLE.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR_SIMPLE / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR_GRID.mkdir(parents=True, exist_ok=True)
    OUT_DIR_SIMPLE.mkdir(parents=True, exist_ok=True)

    for encoder_name in ENCODERS:
        print(f"\nProcessing {encoder_name}")

        examples = collect_examples_for_encoder(encoder_name)

        correct_grid, incorrect_grid = choose_examples(
            examples,
            num_correct=NUM_CORRECT_GRID,
            num_incorrect=NUM_INCORRECT_GRID,
        )

        correct_simple, incorrect_simple = choose_examples(
            examples,
            num_correct=NUM_CORRECT_SIMPLE,
            num_incorrect=NUM_INCORRECT_SIMPLE,
        )

        print(f"  grid correct examples shown:      {len(correct_grid)}")
        print(f"  grid incorrect examples shown:    {len(incorrect_grid)}")
        print(f"  simple correct examples shown:    {len(correct_simple)}")
        print(f"  simple incorrect examples shown:  {len(incorrect_simple)}")

        make_encoder_figure(
            encoder_name,
            correct_grid,
            incorrect_grid,
        )

        make_figure(
            encoder_name,
            correct_simple,
            title="Correctly classified examples",
            out_name=f"{encoder_name}_correct.png",
        )

        make_figure(
            encoder_name,
            incorrect_simple,
            title="Incorrectly classified examples",
            out_name=f"{encoder_name}_incorrect.png",
        )


if __name__ == "__main__":
    main()

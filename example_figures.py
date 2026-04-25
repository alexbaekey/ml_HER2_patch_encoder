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
OUT_DIR = RUNS_DIR / "top_patch_figures"

ENCODERS = [
    "resnet50",
    "vit_base_patch16_224",
    "convnext_base",
]

PATCH_SIZE = 256
TARGET_MAG = 20
NUM_CORRECT = 5
NUM_INCORRECT = 5

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

    def forward(self, bag):
        h = torch.tanh(self.fc(bag))
        h = self.dropout(h)
        a = self.attn(h).squeeze(1)
        w = torch.softmax(a, dim=0)
        z = (w.unsqueeze(1) * h).sum(dim=0)
        logits = self.cls(z)
        return logits, w


def load_model(encoder_name: str):
    ckpt_path = RUNS_DIR / "models" / f"{encoder_name}_best.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_dim, in_dim = state_dict["fc.weight"].shape
    num_classes = state_dict["cls.weight"].shape[0]

    model = ABMIL(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=0.25)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


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


def read_top_patch(mrxs_path: Path, x0: int, y0: int, patch_size: int = 256, target_mag: int = 20):
    slide = openslide.OpenSlide(str(mrxs_path))
    try:
        level = choose_level(slide, target_mag=target_mag)
        img = slide.read_region((int(x0), int(y0)), level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()
    return img


def add_border(img: Image.Image, color: str, width: int = 6):
    return ImageOps.expand(img, border=width, fill=color)


# ============================================================
# LABEL UTILS
# ============================================================

def label_str(y: int) -> str:
    return "Positive" if y == 1 else "Negative"


# ============================================================
# COLLECT TOP-PATCH EXAMPLES
# ============================================================

@torch.no_grad()
def collect_examples_for_encoder(encoder_name: str):
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
        top_idx = int(np.argmax(attn_np))
        top_coord = coords[top_idx].numpy()

        examples.append({
            "slide_id": slide_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "prob_positive": float(probs[1]),
            "correct": pred_label == true_label,
            "num_patches": int(bag.shape[0]),
            "top_x": int(top_coord[0]),
            "top_y": int(top_coord[1]),
        })

    return examples


# ============================================================
# PLOTTING
# ============================================================

def make_figure(encoder_name: str, examples: list[dict], title: str, out_name: str):
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
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
        examples = collect_examples_for_encoder(encoder_name)

        correct = [x for x in examples if x["correct"]]
        incorrect = [x for x in examples if not x["correct"]]

        correct = sorted(correct, key=lambda x: -x["confidence"])[:NUM_CORRECT]
        incorrect = sorted(incorrect, key=lambda x: -x["confidence"])[:NUM_INCORRECT]

        make_figure(
            encoder_name,
            correct,
            title="Correctly classified examples",
            out_name=f"{encoder_name}_correct.png",
        )

        make_figure(
            encoder_name,
            incorrect,
            title="Incorrectly classified examples",
            out_name=f"{encoder_name}_incorrect.png",
        )


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import Counter

import math
import numpy as np
import pandas as pd
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("HEROHE")
TRAIN_XLSX = ROOT / "Training (ground truth).xlsx"
TEST_XLSX = ROOT / "Test (ground truth).xlsx"

OUT_DIR = Path("herohe_data_exploration")

PATCH_SIZE = 256
TARGET_MAG = 20
THUMB_SIZE = 2048
TISSUE_THRESHOLD = 0.85

# This only affects the "sampled_for_model" reporting.
# It does not affect the estimated total number of tissue patches.
MAX_PATCHES_PER_SLIDE = 10000


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clean_case_id(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def label_to_int(x) -> int:
    s = str(x).strip().lower()
    if s == "negative":
        return 0
    if s == "positive":
        return 1
    raise ValueError(f"Unknown label: {x}")


def label_name(y: int) -> str:
    return "Positive" if int(y) == 1 else "Negative"


# ============================================================
# LABEL LOADING
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
            raise ValueError(
                f"Missing column '{c}' in {xlsx_path}. Found: {list(df.columns)}"
            )

    df = df[df["Immunohistochemistry"] == 2].copy()

    labels = {}
    for _, row in df.iterrows():
        sid = clean_case_id(row["Case"])
        y = label_to_int(row["Final Result (Ground truth)"])
        labels[sid] = y

    return labels


def build_slide_table(split_dir: Path, labels: dict[str, int], split_name: str) -> pd.DataFrame:
    rows = []

    for sid, y in labels.items():
        mrxs = split_dir / f"{sid}.mrxs"
        folder = split_dir / sid

        rows.append({
            "slide_id": sid,
            "split": split_name,
            "label": y,
            "label_name": label_name(y),
            "mrxs_path": str(mrxs),
            "folder_path": str(folder),
            "mrxs_exists": mrxs.exists(),
            "folder_exists": folder.exists(),
        })

    return pd.DataFrame(rows)


# ============================================================
# OPENSLIDE HELPERS
# ============================================================

def choose_level(slide: openslide.OpenSlide, target_mag: int = 20) -> int:
    """
    Same best-effort magnification-level selection as the training script.
    """
    props = slide.properties

    if "openslide.mpp-x" in props:
        try:
            mpp = float(props["openslide.mpp-x"])

            # crude mapping:
            # 0.25 um/px ~ 40x
            # 0.50 um/px ~ 20x
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
    """
    Same tissue mask idea as training:
      tissue = mean RGB < 0.85

    Returns:
      tissue_mask: bool thumbnail mask
      mean_rgb: thumbnail mean RGB image
      sx, sy: level-0 scaling from thumbnail pixels
    """
    w0, h0 = slide.dimensions

    scale = min(thumb_size / w0, thumb_size / h0, 1.0)
    tw, th = max(1, int(w0 * scale)), max(1, int(h0 * scale))

    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    arr = np.asarray(thumb).astype(np.float32) / 255.0

    mean_rgb = arr.mean(axis=2)
    tissue_mask = mean_rgb < TISSUE_THRESHOLD

    sx = w0 / tw
    sy = h0 / th

    return tissue_mask, mean_rgb, sx, sy


def estimate_grid_patch_counts(slide: openslide.OpenSlide, level: int, patch_size: int):
    """
    Estimate exhaustive non-overlapping patch grid count at selected OpenSlide level.
    """
    w_level, h_level = slide.level_dimensions[level]

    n_x = math.ceil(w_level / patch_size)
    n_y = math.ceil(h_level / patch_size)

    total_grid_patches = n_x * n_y

    return w_level, h_level, n_x, n_y, total_grid_patches


def estimate_grid_tissue_background_counts(
    slide: openslide.OpenSlide,
    level: int,
    patch_size: int,
    tissue_mask: np.ndarray,
    sx: float,
    sy: float,
):
    """
    Estimate how many full-grid patches are tissue/background using the same
    thumbnail 0.85 threshold.

    Method:
      - make non-overlapping grid patch centers at selected level
      - map each center to level-0 coords
      - map level-0 coords to thumbnail coords
      - check whether thumbnail pixel is tissue
    """
    w_level, h_level = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])

    thumb_h, thumb_w = tissue_mask.shape

    n_x = math.ceil(w_level / patch_size)
    n_y = math.ceil(h_level / patch_size)

    tissue_count = 0
    background_count = 0

    for iy in range(n_y):
        for ix in range(n_x):
            # center in selected level coordinates
            cx_level = min(ix * patch_size + patch_size // 2, w_level - 1)
            cy_level = min(iy * patch_size + patch_size // 2, h_level - 1)

            # convert selected-level center to level-0 coordinate
            cx0 = cx_level * downsample
            cy0 = cy_level * downsample

            # convert level-0 coordinate to thumbnail coordinate
            tx = int(cx0 / sx)
            ty = int(cy0 / sy)

            tx = max(0, min(tx, thumb_w - 1))
            ty = max(0, min(ty, thumb_h - 1))

            if tissue_mask[ty, tx]:
                tissue_count += 1
            else:
                background_count += 1

    return tissue_count, background_count


# ============================================================
# SLIDE ANALYSIS
# ============================================================

def analyze_slide(row) -> dict:
    slide_id = row["slide_id"]
    split = row["split"]
    label = int(row["label"])
    mrxs_path = Path(row["mrxs_path"])

    result = {
        "slide_id": slide_id,
        "split": split,
        "label": label,
        "label_name": label_name(label),
        "mrxs_path": str(mrxs_path),
        "open_ok": False,
        "error": "",
    }

    try:
        slide = openslide.OpenSlide(str(mrxs_path))
    except Exception as e:
        result["error"] = repr(e)
        return result

    try:
        level = choose_level(slide, TARGET_MAG)

        w0, h0 = slide.dimensions

        w_level, h_level, n_x, n_y, total_grid_patches = estimate_grid_patch_counts(
            slide=slide,
            level=level,
            patch_size=PATCH_SIZE,
        )

        tissue_mask, mean_rgb, sx, sy = make_tissue_mask(
            slide=slide,
            thumb_size=THUMB_SIZE,
        )

        thumb_total_locations = int(tissue_mask.size)
        thumb_tissue_locations = int(tissue_mask.sum())
        thumb_background_locations = int(thumb_total_locations - thumb_tissue_locations)

        grid_tissue_patches, grid_background_patches = estimate_grid_tissue_background_counts(
            slide=slide,
            level=level,
            patch_size=PATCH_SIZE,
            tissue_mask=tissue_mask,
            sx=sx,
            sy=sy,
        )

        sampled_for_model = min(MAX_PATCHES_PER_SLIDE, thumb_tissue_locations)

        result.update({
            "open_ok": True,

            # selected WSI level
            "selected_level": int(level),
            "level_downsample": float(slide.level_downsamples[level]),

            # slide sizes
            "level0_width": int(w0),
            "level0_height": int(h0),
            "selected_level_width": int(w_level),
            "selected_level_height": int(h_level),

            # full non-overlapping grid estimate at selected level
            "grid_n_x": int(n_x),
            "grid_n_y": int(n_y),
            "grid_total_patches": int(total_grid_patches),

            # patches kept/dropped by 85% cutoff
            "grid_tissue_patches_below_85": int(grid_tissue_patches),
            "grid_background_patches_above_85": int(grid_background_patches),
            "grid_fraction_tissue_below_85": (
                grid_tissue_patches / total_grid_patches if total_grid_patches > 0 else 0
            ),
            "grid_fraction_background_above_85": (
                grid_background_patches / total_grid_patches if total_grid_patches > 0 else 0
            ),

            # thumbnail-level candidate locations used by your current sampler
            "thumb_total_locations": int(thumb_total_locations),
            "thumb_tissue_locations_below_85": int(thumb_tissue_locations),
            "thumb_background_locations_above_85": int(thumb_background_locations),
            "thumb_fraction_tissue_below_85": (
                thumb_tissue_locations / thumb_total_locations if thumb_total_locations > 0 else 0
            ),
            "thumb_fraction_background_above_85": (
                thumb_background_locations / thumb_total_locations if thumb_total_locations > 0 else 0
            ),

            # what your model would receive at this MAX_PATCHES_PER_SLIDE
            "max_patches_per_slide": int(MAX_PATCHES_PER_SLIDE),
            "sampled_for_model": int(sampled_for_model),
            "not_sampled_from_thumb_tissue_pool": int(max(0, thumb_tissue_locations - sampled_for_model)),
        })

    except Exception as e:
        result["error"] = repr(e)

    finally:
        slide.close()

    return result


# ============================================================
# PRINTED SUMMARY
# ============================================================

def print_summary(df: pd.DataFrame, name: str):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    ok = df[df["open_ok"] == True].copy()
    bad = df[df["open_ok"] == False].copy()

    print("slides total:", len(df))
    print("slides opened:", len(ok))
    print("slides failed:", len(bad))

    if len(ok) == 0:
        return

    print("\nclass counts:")
    print(Counter(ok["label_name"]))

    metrics = [
        "grid_total_patches",
        "grid_tissue_patches_below_85",
        "grid_background_patches_above_85",
        "grid_fraction_tissue_below_85",
        "grid_fraction_background_above_85",
        "thumb_total_locations",
        "thumb_tissue_locations_below_85",
        "thumb_background_locations_above_85",
        "sampled_for_model",
        "not_sampled_from_thumb_tissue_pool",
    ]

    print("\nsummary statistics:")
    print(ok[metrics].describe().T)

    print("\nkey averages:")
    print("avg full-grid patches per WSI:",
          ok["grid_total_patches"].mean())
    print("avg full-grid patches kept below 85% threshold:",
          ok["grid_tissue_patches_below_85"].mean())
    print("avg full-grid patches dropped/noisy above 85% threshold:",
          ok["grid_background_patches_above_85"].mean())
    print("avg fraction kept below 85% threshold:",
          ok["grid_fraction_tissue_below_85"].mean())
    print("avg fraction dropped/noisy above 85% threshold:",
          ok["grid_fraction_background_above_85"].mean())
    print("avg model-sampled patches per WSI:",
          ok["sampled_for_model"].mean())

    max_row = ok.loc[ok["grid_total_patches"].idxmax()]
    min_row = ok.loc[ok["grid_total_patches"].idxmin()]

    print("\nmax-patch slide:")
    print(max_row[[
        "split",
        "slide_id",
        "label_name",
        "grid_total_patches",
        "grid_tissue_patches_below_85",
        "grid_background_patches_above_85",
        "grid_fraction_tissue_below_85",
    ]])

    print("\nmin-patch slide:")
    print(min_row[[
        "split",
        "slide_id",
        "label_name",
        "grid_total_patches",
        "grid_tissue_patches_below_85",
        "grid_background_patches_above_85",
        "grid_fraction_tissue_below_85",
    ]])


# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, fname: str, bins: int = 30):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(ok[col].dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of WSIs")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_hist_by_label(df: pd.DataFrame, col: str, title: str, xlabel: str, fname: str, bins: int = 30):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    plt.figure(figsize=(8, 5))

    for lab in ["Negative", "Positive"]:
        sub = ok[ok["label_name"] == lab]
        if len(sub) > 0:
            plt.hist(sub[col].dropna().values, bins=bins, alpha=0.5, label=lab)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of WSIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_hist_by_split(df: pd.DataFrame, col: str, title: str, xlabel: str, fname: str, bins: int = 30):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    plt.figure(figsize=(8, 5))

    for split in ["train", "test"]:
        sub = ok[ok["split"] == split]
        if len(sub) > 0:
            plt.hist(sub[col].dropna().values, bins=bins, alpha=0.5, label=split)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of WSIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_box_by_label(df: pd.DataFrame, col: str, title: str, ylabel: str, fname: str):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    data = []
    labels = []

    for lab in ["Negative", "Positive"]:
        sub = ok[ok["label_name"] == lab]
        if len(sub) > 0:
            data.append(sub[col].dropna().values)
            labels.append(lab)

    if len(data) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_box_by_split(df: pd.DataFrame, col: str, title: str, ylabel: str, fname: str):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    data = []
    labels = []

    for split in ["train", "test"]:
        sub = ok[ok["split"] == split]
        if len(sub) > 0:
            data.append(sub[col].dropna().values)
            labels.append(split)

    if len(data) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str, fname: str):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    plt.figure(figsize=(7, 5))

    for lab in ["Negative", "Positive"]:
        sub = ok[ok["label_name"] == lab]
        if len(sub) > 0:
            plt.scatter(sub[xcol].values, sub[ycol].values, alpha=0.7, label=lab)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def save_bar_means(df: pd.DataFrame):
    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        return

    cols = [
        "grid_total_patches",
        "grid_tissue_patches_below_85",
        "grid_background_patches_above_85",
        "sampled_for_model",
    ]

    means = ok[cols].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cols)), means.values)
    plt.xticks(range(len(cols)), cols, rotation=30, ha="right")
    plt.ylabel("Average count per WSI")
    plt.title("Average patch-related counts per WSI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "average_patch_counts_bar.png", dpi=200)
    plt.close()


def save_distribution_plots(df: pd.DataFrame):
    ensure_dir(OUT_DIR)

    ok = df[df["open_ok"] == True].copy()
    if len(ok) == 0:
        print("No successfully opened slides. Skipping plots.")
        return

    # ------------------------------------------------------------
    # Main histograms
    # ------------------------------------------------------------

    save_hist(
        ok,
        "grid_total_patches",
        "Distribution of estimated full-grid patches per WSI",
        "Estimated full-grid patches per WSI",
        "hist_grid_total_patches.png",
    )

    save_hist(
        ok,
        "grid_tissue_patches_below_85",
        "Distribution of estimated tissue patches kept by 85% cutoff",
        "Estimated tissue patches below 85% brightness cutoff",
        "hist_grid_tissue_patches_below_85.png",
    )

    save_hist(
        ok,
        "grid_background_patches_above_85",
        "Distribution of estimated patches dropped by 85% cutoff",
        "Estimated bright/background patches above 85% cutoff",
        "hist_grid_background_patches_above_85.png",
    )

    save_hist(
        ok,
        "grid_fraction_tissue_below_85",
        "Distribution of fraction of patches kept by 85% cutoff",
        "Fraction of full-grid patches kept",
        "hist_grid_fraction_tissue_below_85.png",
    )

    save_hist(
        ok,
        "grid_fraction_background_above_85",
        "Distribution of fraction of patches dropped by 85% cutoff",
        "Fraction of full-grid patches dropped",
        "hist_grid_fraction_background_above_85.png",
    )

    save_hist(
        ok,
        "sampled_for_model",
        "Distribution of sampled patches used by model",
        "Sampled patches per WSI",
        "hist_sampled_for_model.png",
    )

    # ------------------------------------------------------------
    # By label histograms
    # ------------------------------------------------------------

    save_hist_by_label(
        ok,
        "grid_total_patches",
        "Full-grid patches per WSI by class",
        "Estimated full-grid patches per WSI",
        "hist_by_label_grid_total_patches.png",
    )

    save_hist_by_label(
        ok,
        "grid_tissue_patches_below_85",
        "Tissue patches kept by 85% cutoff by class",
        "Estimated tissue patches below 85% brightness cutoff",
        "hist_by_label_grid_tissue_below_85.png",
    )

    save_hist_by_label(
        ok,
        "grid_background_patches_above_85",
        "Patches dropped by 85% cutoff by class",
        "Estimated bright/background patches above 85% cutoff",
        "hist_by_label_grid_background_above_85.png",
    )

    save_hist_by_label(
        ok,
        "grid_fraction_tissue_below_85",
        "Fraction of patches kept by 85% cutoff by class",
        "Fraction of full-grid patches kept",
        "hist_by_label_grid_fraction_tissue_below_85.png",
    )

    # ------------------------------------------------------------
    # By split histograms
    # ------------------------------------------------------------

    save_hist_by_split(
        ok,
        "grid_total_patches",
        "Full-grid patches per WSI by split",
        "Estimated full-grid patches per WSI",
        "hist_by_split_grid_total_patches.png",
    )

    save_hist_by_split(
        ok,
        "grid_tissue_patches_below_85",
        "Tissue patches kept by 85% cutoff by split",
        "Estimated tissue patches below 85% brightness cutoff",
        "hist_by_split_grid_tissue_below_85.png",
    )

    save_hist_by_split(
        ok,
        "grid_background_patches_above_85",
        "Patches dropped by 85% cutoff by split",
        "Estimated bright/background patches above 85% cutoff",
        "hist_by_split_grid_background_above_85.png",
    )

    # ------------------------------------------------------------
    # Boxplots
    # ------------------------------------------------------------

    save_box_by_label(
        ok,
        "grid_total_patches",
        "Full-grid patch count by class",
        "Estimated full-grid patches per WSI",
        "box_by_label_grid_total_patches.png",
    )

    save_box_by_label(
        ok,
        "grid_tissue_patches_below_85",
        "Tissue patch count below 85% cutoff by class",
        "Estimated tissue patches",
        "box_by_label_grid_tissue_below_85.png",
    )

    save_box_by_label(
        ok,
        "grid_fraction_tissue_below_85",
        "Fraction of patches kept by 85% cutoff by class",
        "Fraction kept",
        "box_by_label_grid_fraction_tissue_below_85.png",
    )

    save_box_by_split(
        ok,
        "grid_total_patches",
        "Full-grid patch count by train/test split",
        "Estimated full-grid patches per WSI",
        "box_by_split_grid_total_patches.png",
    )

    save_box_by_split(
        ok,
        "grid_tissue_patches_below_85",
        "Tissue patch count below 85% cutoff by train/test split",
        "Estimated tissue patches",
        "box_by_split_grid_tissue_below_85.png",
    )

    # ------------------------------------------------------------
    # Scatter plots
    # ------------------------------------------------------------

    save_scatter(
        ok,
        "grid_total_patches",
        "grid_tissue_patches_below_85",
        "Total patches vs. tissue patches kept",
        "Estimated full-grid patches",
        "Estimated tissue patches below 85% cutoff",
        "scatter_total_vs_tissue_kept.png",
    )

    save_scatter(
        ok,
        "grid_total_patches",
        "grid_background_patches_above_85",
        "Total patches vs. patches dropped by cutoff",
        "Estimated full-grid patches",
        "Estimated bright/background patches above 85% cutoff",
        "scatter_total_vs_background_dropped.png",
    )

    save_scatter(
        ok,
        "grid_fraction_tissue_below_85",
        "grid_tissue_patches_below_85",
        "Fraction kept vs. tissue patch count",
        "Fraction kept by 85% cutoff",
        "Estimated tissue patches below 85% cutoff",
        "scatter_fraction_kept_vs_tissue_count.png",
    )

    # ------------------------------------------------------------
    # Mean bar chart
    # ------------------------------------------------------------

    save_bar_means(ok)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    train_labels = load_ihc2_binary_labels(TRAIN_XLSX)
    test_labels = load_ihc2_binary_labels(TEST_XLSX)

    train_df = build_slide_table(ROOT / "train", train_labels, "train")
    test_df = build_slide_table(ROOT / "test", test_labels, "test")

    all_df = pd.concat([train_df, test_df], ignore_index=True)

    print("Train labels:", Counter(train_labels.values()))
    print("Test labels:", Counter(test_labels.values()))
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("All rows:", len(all_df))

    existing_df = all_df[
        (all_df["mrxs_exists"] == True) &
        (all_df["folder_exists"] == True)
    ].copy()

    missing_df = all_df[
        (all_df["mrxs_exists"] == False) |
        (all_df["folder_exists"] == False)
    ].copy()

    if len(missing_df) > 0:
        missing_path = OUT_DIR / "missing_slides.csv"
        missing_df.to_csv(missing_path, index=False)
        print(f"\nMissing slide entries saved to: {missing_path}")

    rows = []
    for _, row in tqdm(
        existing_df.iterrows(),
        total=len(existing_df),
        desc="Analyzing HEROHE WSIs",
    ):
        rows.append(analyze_slide(row))

    stats_df = pd.DataFrame(rows)

    stats_path = OUT_DIR / "herohe_patch_statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    print_summary(stats_df[stats_df["split"] == "train"], "TRAIN SUMMARY")
    print_summary(stats_df[stats_df["split"] == "test"], "TEST SUMMARY")
    print_summary(stats_df, "ALL SUMMARY")

    print("\nSaving distribution plots...")
    save_distribution_plots(stats_df)

    print("\nSaved CSV:")
    print(stats_path)

    print("\nSaved plots to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()

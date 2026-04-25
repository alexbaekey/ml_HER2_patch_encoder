# filter_herohe_ihc2.py
#
# Keep ONLY cases with Immunohistochemistry == 2
# For both:
#   1. Excel sheets
#   2. Actual WSI folders/files (.mrxs + numbered dirs)
#
# Usage:
#   python filter_herohe_ihc2.py
#
# Edit ROOT below if needed.

from pathlib import Path
import pandas as pd
import shutil

# =====================================================
# SETTINGS
# =====================================================

ROOT = Path("HEROHE")

TRAIN_XLSX = ROOT / "Training (ground truth).xlsx"
TEST_XLSX  = ROOT / "Test (ground truth).xlsx"

OUT_ROOT = Path("HEROHE_IHC2")

COPY_FILES = True      # True = copy data
                      # False = move data

# =====================================================
# HELPERS
# =====================================================

def copy_or_move(src, dst):
    if COPY_FILES:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def clean_case_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def filter_sheet(xlsx_path, out_xlsx):
    """
    Keep only rows where Immunohistochemistry == 2
    """
    df = pd.read_excel(xlsx_path)

    df = df[df["Immunohistochemistry"] == 2].copy()

    df.to_excel(out_xlsx, index=False)

    case_ids = set(df["Case"].apply(clean_case_id).tolist())

    return case_ids


def copy_cases(split_name, case_ids):
    """
    Copies:
      train/63 folder
      train/63.mrxs file
    """
    src_split = ROOT / split_name
    dst_split = OUT_ROOT / split_name
    dst_split.mkdir(parents=True, exist_ok=True)

    for cid in sorted(case_ids):
        folder = src_split / cid
        mrxs   = src_split / f"{cid}.mrxs"

        if folder.exists():
            print("copy", folder)
            copy_or_move(folder, dst_split / cid)

        if mrxs.exists():
            print("copy", mrxs)
            copy_or_move(mrxs, dst_split / f"{cid}.mrxs")


# =====================================================
# MAIN
# =====================================================

def main():
    OUT_ROOT.mkdir(exist_ok=True)

    # -----------------------------
    # TRAIN
    # -----------------------------
    train_ids = filter_sheet(
        TRAIN_XLSX,
        OUT_ROOT / "Training (ground truth).xlsx"
    )

    print("Train IHC=2 cases:", len(train_ids))

    copy_cases("train", train_ids)

    # -----------------------------
    # TEST
    # -----------------------------
    if TEST_XLSX.exists():
        test_ids = filter_sheet(
            TEST_XLSX,
            OUT_ROOT / "Test (ground truth).xlsx"
        )

        print("Test IHC=2 cases:", len(test_ids))

        copy_cases("test", test_ids)

    print("\nDone.")
    print("Filtered dataset saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
OUT_DIR = Path("result_bar_plots")
OUT_DIR.mkdir(exist_ok=True)

ENCODER_RUN_PREFIX = "herohe_ihc2_runs_"
AGG_RUN_PREFIX = "herohe_ihc2_aggregation_runs_"

METRICS = [
    "accuracy",
    "macro_f1",
    "auroc",
]

SPLIT = "test"   # use "test" or "val"


# ============================================================
# HELPERS
# ============================================================

def extract_patch_count_from_dirname(dirname: str) -> int | None:
    """
    Examples:
      herohe_ihc2_runs_5 -> 5
      herohe_ihc2_runs_1000 -> 1000
      herohe_ihc2_aggregation_runs_100patch -> 100
      herohe_ihc2_aggregation_runs_1000patch -> 1000
    """
    nums = re.findall(r"\d+", dirname)
    if not nums:
        return None
    return int(nums[-1])


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def safe_metric(d: dict, split: str, metric: str):
    try:
        return d[split][metric]
    except KeyError:
        return np.nan


# ============================================================
# LOAD ENCODER RESULTS
# ============================================================

def load_encoder_results() -> pd.DataFrame:
    rows = []

    run_dirs = sorted([
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith(ENCODER_RUN_PREFIX)
    ])

    for run_dir in run_dirs:
        patch_count = extract_patch_count_from_dirname(run_dir.name)
        if patch_count is None:
            continue

        result_files = sorted(run_dir.glob("results_*.json"))

        for result_file in result_files:
            d = load_json(result_file)

            encoder = d.get("encoder", result_file.stem.replace("results_", ""))

            row = {
                "patches": patch_count,
                "encoder": encoder,
                "result_file": str(result_file),
            }

            for metric in METRICS:
                row[metric] = safe_metric(d, SPLIT, metric)

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# LOAD AGGREGATION RESULTS
# ============================================================

def load_aggregation_results() -> pd.DataFrame:
    rows = []

    run_dirs = sorted([
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith(AGG_RUN_PREFIX)
    ])

    for run_dir in run_dirs:
        patch_count = extract_patch_count_from_dirname(run_dir.name)
        if patch_count is None:
            continue

        result_files = sorted(run_dir.glob("results_*.json"))

        for result_file in result_files:
            d = load_json(result_file)

            aggregator = d.get("aggregator", result_file.stem.replace("results_ctranspath_", ""))

            row = {
                "patches": patch_count,
                "aggregator": aggregator,
                "result_file": str(result_file),
            }

            for metric in METRICS:
                row[metric] = safe_metric(d, SPLIT, metric)

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def grouped_bar_plot(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    if df.empty:
        print(f"No data for {title}")
        return

    plot_df = df.pivot_table(
        index="patches",
        columns=group_col,
        values=metric,
        aggfunc="mean",
    )

    plot_df = plot_df.sort_index()

    x_labels = [str(x) for x in plot_df.index.tolist()]
    groups = plot_df.columns.tolist()

    x = np.arange(len(x_labels))
    width = 0.8 / max(1, len(groups))

    plt.figure(figsize=(11, 6))

    for i, group in enumerate(groups):
        values = plot_df[group].values
        offset = (i - (len(groups) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=group)

    plt.xlabel("MAX_PATCHES_PER_SLIDE")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, x_labels)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved: {out_path}")


def make_all_plots(df: pd.DataFrame, group_col: str, name: str):
    if df.empty:
        print(f"No {name} results found.")
        return

    df = df.copy()
    df = df.sort_values(["patches", group_col])

    csv_path = OUT_DIR / f"{name}_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

    for metric in METRICS:
        if metric not in df.columns:
            continue

        grouped_bar_plot(
            df=df,
            group_col=group_col,
            metric=metric,
            title=f"{name}: {SPLIT} {metric}",
            ylabel=f"{SPLIT} {metric}",
            out_path=OUT_DIR / f"{name}_{SPLIT}_{metric}.png",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    encoder_df = load_encoder_results()
    aggregation_df = load_aggregation_results()

    print("\nEncoder results:")
    print(encoder_df)

    print("\nAggregation results:")
    print(aggregation_df)

    make_all_plots(
        df=encoder_df,
        group_col="encoder",
        name="encoder",
    )

    make_all_plots(
        df=aggregation_df,
        group_col="aggregator",
        name="aggregation",
    )


if __name__ == "__main__":
    main()

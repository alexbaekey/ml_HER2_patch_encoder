from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
OUT_DIR = Path("reported_metrics")
OUT_DIR.mkdir(exist_ok=True)

SPLIT = "test"   # use "test" or "val"

METRICS = [
    "auc",
    "precision",
    "recall",
    "f1_score",
]


# ============================================================
# HELPERS
# ============================================================

def extract_patch_count(dirname: str):
    """
    Examples:
      herohe_ihc2_runs_100 -> 100
      herohe_ihc2_runs_10000 -> 10000
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


def metrics_from_confusion_matrix(cm):
    """
    Assumes confusion matrix format:

        [[TN, FP],
         [FN, TP]]

    Computes positive-class precision, recall, and F1.
    """
    cm = np.array(cm)

    if cm.shape != (2, 2):
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        }

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def parse_result_file(result_file: Path, run_type: str, patch_count: int):
    d = load_json(result_file)

    split_data = d.get(SPLIT, {})
    cm = split_data.get("confusion_matrix", None)

    if cm is None:
        precision = np.nan
        recall = np.nan
        f1_score = np.nan
    else:
        m = metrics_from_confusion_matrix(cm)
        precision = m["precision"]
        recall = m["recall"]
        f1_score = m["f1_score"]

    if run_type == "encoder":
        method = d.get("encoder", result_file.stem.replace("results_", ""))
    else:
        method = d.get("aggregator", result_file.stem.replace("results_ctranspath_", ""))

    return {
        "run_type": run_type,
        "patches": patch_count,
        "method": method,
        "auc": split_data.get("auroc", np.nan),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "result_file": str(result_file),
    }


# ============================================================
# COLLECT RESULTS
# ============================================================

def collect_encoder_results():
    rows = []

    for run_dir in sorted(ROOT.glob("herohe_ihc2_runs_*")):
        if not run_dir.is_dir():
            continue

        patch_count = extract_patch_count(run_dir.name)
        if patch_count is None:
            continue

        for result_file in sorted(run_dir.glob("results_*.json")):
            rows.append(parse_result_file(result_file, "encoder", patch_count))

    return pd.DataFrame(rows)


def collect_aggregation_results():
    rows = []

    for run_dir in sorted(ROOT.glob("herohe_ihc2_aggregation_runs_*")):
        if not run_dir.is_dir():
            continue

        patch_count = extract_patch_count(run_dir.name)
        if patch_count is None:
            continue

        for result_file in sorted(run_dir.glob("results_*.json")):
            rows.append(parse_result_file(result_file, "aggregation", patch_count))

    return pd.DataFrame(rows)


# ============================================================
# REPORT TABLES
# ============================================================

def print_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("No results found.")
        return

    cols = [
        "patches",
        "method",
        "auc",
        "precision",
        "recall",
        "f1_score",
    ]

    df_show = df[cols].copy()
    df_show = df_show.sort_values(["patches", "method"])

    print(df_show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_table(df: pd.DataFrame, filename: str):
    if df.empty:
        return

    out_path = OUT_DIR / filename
    df = df.sort_values(["patches", "method"])
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


# ============================================================
# BAR PLOTS
# ============================================================

def grouped_bar_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
):
    if df.empty:
        print(f"No data for {title}")
        return

    plot_df = df.pivot_table(
        index="patches",
        columns="method",
        values=metric,
        aggfunc="mean",
    )

    plot_df = plot_df.sort_index()

    x_labels = [str(x) for x in plot_df.index.tolist()]
    methods = plot_df.columns.tolist()

    x = np.arange(len(x_labels))
    width = 0.8 / max(1, len(methods))

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        values = plot_df[method].values
        offset = (i - (len(methods) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=method)

    plt.xlabel("MAX_PATCHES_PER_SLIDE")
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(x, x_labels)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved: {out_path}")


def make_metric_plots(df: pd.DataFrame, name: str):
    if df.empty:
        print(f"No {name} results found.")
        return

    for metric in METRICS:
        grouped_bar_plot(
            df=df,
            metric=metric,
            title=f"{name} {SPLIT} {metric}",
            out_path=OUT_DIR / f"{name}_{SPLIT}_{metric}.png",
        )


def make_combined_metric_plot(df: pd.DataFrame, name: str):
    """
    One combined figure per result type.
    Subplots: AUC, Precision, Recall, F1.
    """
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        plot_df = df.pivot_table(
            index="patches",
            columns="method",
            values=metric,
            aggfunc="mean",
        ).sort_index()

        x_labels = [str(x) for x in plot_df.index.tolist()]
        methods = plot_df.columns.tolist()

        x = np.arange(len(x_labels))
        width = 0.8 / max(1, len(methods))

        for i, method in enumerate(methods):
            values = plot_df[method].values
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=method)

        ax.set_title(metric)
        ax.set_xlabel("MAX_PATCHES_PER_SLIDE")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))

    fig.suptitle(f"{name} {SPLIT} metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_path = OUT_DIR / f"{name}_{SPLIT}_combined_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    encoder_df = collect_encoder_results()
    aggregation_df = collect_aggregation_results()

    print_table(
        encoder_df,
        f"ENCODER RESULTS ({SPLIT})"
    )

    print_table(
        aggregation_df,
        f"AGGREGATION RESULTS ({SPLIT})"
    )

    save_table(
        encoder_df,
        f"encoder_{SPLIT}_metrics.csv"
    )

    save_table(
        aggregation_df,
        f"aggregation_{SPLIT}_metrics.csv"
    )

    make_metric_plots(
        encoder_df,
        "encoder"
    )

    make_metric_plots(
        aggregation_df,
        "aggregation"
    )

    make_combined_metric_plot(
        encoder_df,
        "encoder"
    )

    make_combined_metric_plot(
        aggregation_df,
        "aggregation"
    )


if __name__ == "__main__":
    main()

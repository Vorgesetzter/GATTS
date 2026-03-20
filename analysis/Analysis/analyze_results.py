"""
Analyze adversarial TTS experiment results.

Graphs produced:
  1. convergence_bar.png        — generation distribution of successful runs
  2. cumulative_success.png     — cumulative success rate over generations
  3. success_pie.png            — overall success vs failure
  4. sentence_outcomes.png      — per-sentence 0/1/2/3 success distribution
  5. failed_proximity.png       — failed runs proximity to thresholds
  6. convergence_curve_all.png  — avg best PESQ/SET_OVERLAP per generation (all runs)
  7. convergence_curve_failed.png — same for failed runs only
  8. example_run.png            — composite of graphs for the median-convergence run
  9. comparison_table.png       — GT text vs ASR transcription for successful runs
 10. semantic_similarity_all.png — cosine similarity for every run, coloured by success
 11. iqr_boxplots.png           — IQR box plots for semantic_similarity, SET_OVERLAP, PESQ
 12. utmos_scores.png          — UTMOS MOS scores: GT vs adversarial, grouped by success/failure

Usage:
    python Scripts/Analysis/analyze_results.py outputs/results/20260302_1006
    python Scripts/Analysis/analyze_results.py outputs/results/20260302_1006 \\
        --output_dir outputs/analysis --example_run 51:0
"""

import os
import re
import json
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
_STOPWORDS  = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def _lemmatize_word(word: str) -> str:
    for pos in ("a", "v", "n", "r"):
        lemma = _LEMMATIZER.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


def _set_overlap(gt: str, asr: str) -> float:
    clean_gt  = re.sub(r"[^\w\s]", "", gt.lower())
    gt_words  = {_lemmatize_word(w) for w in set(clean_gt.split()) - _STOPWORDS}
    if not gt_words:
        return 1.0
    clean_asr = re.sub(r"[^\w\s]", "", (asr or "").lower())
    asr_words = {_lemmatize_word(w) for w in set(clean_asr.split()) - _STOPWORDS}
    return min(len(gt_words & asr_words) / len(gt_words), 1.0)


PESQ_THRESHOLD        = 0.2
SET_OVERLAP_THRESHOLD = 0.5

COLOR_SUCCESS = "#2ecc71"
COLOR_FAIL    = "#e74c3c"
COLOR_PARTIAL = "#f39c12"
COLOR_PESQ    = "#3498db"
COLOR_SO      = "#9b59b6"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    records = []
    for json_path in sorted(Path(results_dir).rglob("run_summary.json")):
        with open(json_path) as f:
            summary = json.load(f)

        meta     = summary.get("metadata", {})
        text     = summary.get("text_data", {})
        success  = summary.get("success_metrics", {})
        eff      = summary.get("efficiency_metrics", {})
        algo     = summary.get("algorithm_parameters", {})
        solution = summary.get("final_solution", {})
        nat      = summary.get("naturalness_scores", {})

        records.append({
            "sentence_id":          meta.get("sentence_id"),
            "run_id":               meta.get("run_id"),
            "run_timestamp":        meta.get("run_timestamp"),
            "json_path":            str(json_path),
            "ground_truth_text":    text.get("ground_truth_text", ""),
            "asr_transcription":    text.get("asr_transcription", ""),
            "semantic_similarity":  text.get("semantic_similarity"),
            "success":              bool(success.get("success", False)),
            "pesq":                 success.get("fitness_scores", {}).get("PESQ"),
            "set_overlap":          success.get("fitness_scores", {}).get("SET_OVERLAP"),
            "set_overlap_recomputed": _set_overlap(
                text.get("ground_truth_text", ""),
                text.get("asr_transcription", ""),
            ),
            "generation_count":     eff.get("generation_count"),
            "elapsed_time_seconds": eff.get("elapsed_time_seconds"),
            "generation_found":     solution.get("generation_found"),
            "pop_size":             algo.get("pop_size"),
            "num_generations":      algo.get("num_generations"),
            "gt_rms":               algo.get("gt_rms"),
            "target_rms":           algo.get("target_rms"),
            "utmos_best":           nat.get("utmos_best"),
            "utmos_gt":             nat.get("utmos_gt"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(["sentence_id", "run_id"]).reset_index(drop=True)
    df["set_overlap_mismatch"] = (
        df["set_overlap"].notna() &
        (df["set_overlap"] - df["set_overlap_recomputed"]).abs().gt(0.05)
    )
    return df


def load_fitness_histories(results_dir: str, df: pd.DataFrame) -> dict:
    """Returns dict mapping (sentence_id, run_id) -> fitness_history DataFrame."""
    success_lookup = df.set_index(["sentence_id", "run_id"])["success"].to_dict()
    histories = {}

    for json_path in sorted(Path(results_dir).rglob("run_summary.json")):
        fitness_path = json_path.parent / "fitness_history.csv"
        if not fitness_path.exists():
            continue
        try:
            hist_df = pd.read_csv(fitness_path)
        except Exception:
            continue

        sentence_id, run_id = None, None
        for part in json_path.parts:
            if part.startswith("sentence_"):
                try: sentence_id = int(part.split("_")[1])
                except ValueError: pass
            elif part.startswith("run_"):
                try: run_id = int(part.split("_")[1])
                except ValueError: pass

        if sentence_id is not None and run_id is not None:
            key = (sentence_id, run_id)
            hist_df["success"] = success_lookup.get(key, False)
            histories[key] = hist_df

    return histories


def _max_gen(df: pd.DataFrame) -> int:
    return int(df["num_generations"].dropna().max()) if df["num_generations"].notna().any() else 100


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    total_runs      = len(df)
    total_sentences = df["sentence_id"].nunique()
    n_success       = int(df["success"].sum())
    runs_per        = df.groupby("sentence_id")["run_id"].count().max()

    s = df.groupby("sentence_id")["success"].agg(["sum", "count"])
    at_least_one = int((s["sum"] >= 1).sum())
    majority     = int((s["sum"] >= s["count"] * 2 / 3).sum())
    all_success  = int((s["sum"] == s["count"]).sum())
    all_failed   = int((s["sum"] == 0).sum())

    gen_found = df["generation_found"].dropna().astype(int)

    print(f"\n{'='*55}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*55}")
    print(f"  Sentences evaluated:      {total_sentences}")
    print(f"  Runs per sentence:        {runs_per}")
    print(f"  Total runs:               {total_runs}")
    print(f"  Successful runs:          {n_success} / {total_runs}  ({100*n_success/total_runs:.1f}%)")
    print(f"  Failed runs:              {total_runs-n_success} / {total_runs}  ({100*(total_runs-n_success)/total_runs:.1f}%)")
    print(f"\n  Per-sentence breakdown:")
    print(f"    At least 1 success:     {at_least_one:3d} / {total_sentences}  ({100*at_least_one/total_sentences:.1f}%)")
    print(f"    ≥2/3 runs succeeded:    {majority:3d} / {total_sentences}  ({100*majority/total_sentences:.1f}%)")
    print(f"    All runs succeeded:     {all_success:3d} / {total_sentences}  ({100*all_success/total_sentences:.1f}%)")
    print(f"    All runs failed:        {all_failed:3d} / {total_sentences}  ({100*all_failed/total_sentences:.1f}%)")

    if len(gen_found) > 0:
        print(f"\n  Convergence (successful runs only):")
        print(f"    Mean generations:     {gen_found.mean():.1f}")
        print(f"    Median generations:   {gen_found.median():.1f}")
        print(f"    Min / Max:            {gen_found.min()} / {gen_found.max()}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1 — Convergence bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_bar(df: pd.DataFrame, output_dir: str):
    gen_found = df["generation_found"].dropna().astype(int)
    if gen_found.empty:
        print("[Skip] No successful runs — skipping convergence bar chart.")
        return

    max_gen = _max_gen(df)
    counts  = gen_found.value_counts().reindex(range(1, max_gen + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color=COLOR_PESQ,
           edgecolor="white", linewidth=0.6, width=0.85)
    ax.axvline(gen_found.mean(),   color=COLOR_FAIL, linestyle="--", linewidth=2,
               label=f"Mean: {gen_found.mean():.1f}")
    ax.axvline(gen_found.median(), color=COLOR_PARTIAL, linestyle="--", linewidth=2,
               label=f"Median: {gen_found.median():.1f}")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Number of Runs", fontsize=12)
    ax.set_title("Generation at Which Runs Succeeded", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(0, max_gen + 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_bar.png"), dpi=200, bbox_inches="tight")
    print("[Saved] convergence_bar.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2 — Cumulative success curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_success(df: pd.DataFrame, output_dir: str):
    total_runs = len(df)
    max_gen    = _max_gen(df)
    gen_found  = df["generation_found"].dropna().astype(int)

    gens       = np.arange(1, max_gen + 1)
    cumulative = np.array([(gen_found <= g).sum() / total_runs * 100 for g in gens])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gens, cumulative, color=COLOR_SUCCESS, linewidth=2.5)
    ax.fill_between(gens, cumulative, alpha=0.15, color=COLOR_SUCCESS)
    ax.axhline(cumulative[-1], color=COLOR_FAIL, linestyle="--", linewidth=1.5,
               label=f"Final: {cumulative[-1]:.1f}%")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Cumulative Success Rate (%)", fontsize=12)
    ax.set_title("Cumulative Attack Success Rate over Generations", fontsize=14, fontweight="bold")
    ax.set_xlim(1, max_gen)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_success.png"), dpi=200, bbox_inches="tight")
    print("[Saved] cumulative_success.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 3 — Overall success/failure pie
# ─────────────────────────────────────────────────────────────────────────────

def plot_success_pie(df: pd.DataFrame, output_dir: str):
    n_success = int(df["success"].sum())
    n_fail    = len(df) - n_success

    fig, ax = plt.subplots(figsize=(7, 7))
    _, _, autotexts = ax.pie(
        [n_success, n_fail],
        labels=[f"Success\n({n_success})", f"Failed\n({n_fail})"],
        colors=[COLOR_SUCCESS, COLOR_FAIL],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 13},
    )
    for at in autotexts:
        at.set_fontsize(13)

    ax.set_title("Overall Attack Success Rate", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_pie.png"), dpi=200, bbox_inches="tight")
    print("[Saved] success_pie.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 4 — Per-sentence outcome pie
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentence_outcome_pie(df: pd.DataFrame, output_dir: str):
    s        = df.groupby("sentence_id")["success"].agg(["sum", "count"]).reset_index()
    runs_per = int(s["count"].max())
    outcome_counts = s["sum"].value_counts().reindex(range(runs_per + 1), fill_value=0)

    palette = [COLOR_FAIL, COLOR_PARTIAL, "#f1c40f", COLOR_SUCCESS]
    labels, sizes, colors = [], [], []

    for k in range(runs_per + 1):
        n = int(outcome_counts[k])
        if n == 0:
            continue
        if k == 0:
            label = f"0 / {runs_per} succeeded\n({n} sentences)"
        elif k == runs_per:
            label = f"All {runs_per} / {runs_per} succeeded\n({n} sentences)"
        else:
            label = f"{k} / {runs_per} succeeded\n({n} sentences)"
        labels.append(label)
        sizes.append(n)
        colors.append(palette[min(k, len(palette) - 1)])

    if not sizes:
        print("[Skip] No sentence outcome data.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    _, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(12)

    ax.set_title(f"Per-Sentence Outcome Distribution\n(out of {runs_per} runs each)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentence_outcomes.png"), dpi=200, bbox_inches="tight")
    print("[Saved] sentence_outcomes.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 5 — Failed runs proximity scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_failed_proximity(df: pd.DataFrame, output_dir: str):
    failed  = df[~df["success"]].dropna(subset=["pesq", "set_overlap"])
    success = df[df["success"]].dropna(subset=["pesq", "set_overlap"])

    if failed.empty:
        print("[Skip] No failed runs — skipping proximity scatter.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    # Success zone
    ax.add_patch(mpatches.Rectangle(
        (0, 0), PESQ_THRESHOLD, SET_OVERLAP_THRESHOLD,
        linewidth=0, facecolor=COLOR_SUCCESS, alpha=0.08, zorder=1, label="Success zone",
    ))

    ax.scatter(failed["pesq"], failed["set_overlap"],
               color=COLOR_FAIL, s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
               label=f"Failed runs ({len(failed)})", zorder=3)

    if not success.empty:
        ax.scatter(success["pesq"], success["set_overlap"],
                   color=COLOR_SUCCESS, s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                   label=f"Successful runs ({len(success)})", zorder=3)

    ax.axvline(PESQ_THRESHOLD, color=COLOR_PESQ, linestyle="--", linewidth=1.8,
               label=f"PESQ threshold ({PESQ_THRESHOLD})")
    ax.axhline(SET_OVERLAP_THRESHOLD, color=COLOR_SO, linestyle="--", linewidth=1.8,
               label=f"SET_OVERLAP threshold ({SET_OVERLAP_THRESHOLD})")

    ax.set_xlabel("PESQ (lower = better)", fontsize=12)
    ax.set_ylabel("SET_OVERLAP (lower = better)", fontsize=12)
    ax.set_title("Failed Runs — Proximity to Success Thresholds", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "failed_proximity.png"), dpi=200, bbox_inches="tight")
    print("[Saved] failed_proximity.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graphs 6 & 7 — Average convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def _build_curve(hist_list: list, cols: list) -> pd.DataFrame:
    """Average cols over generation, across histories of different lengths."""
    if not hist_list:
        return pd.DataFrame()
    combined = pd.concat(hist_list, ignore_index=True)
    return combined.groupby("generation")[cols].agg(["mean", "std"])


def _draw_curve(ax, curve_df, col, color, label):
    gens = curve_df.index
    mean = curve_df[(col, "mean")]
    std  = curve_df[(col, "std")].fillna(0)
    ax.plot(gens, mean, color=color, linewidth=2.5, label=label)
    ax.fill_between(gens, (mean - std).clip(0), (mean + std).clip(0, 1),
                    color=color, alpha=0.15)


def plot_convergence_curves(histories: dict, df: pd.DataFrame, output_dir: str):
    success_keys = {k for k, v in histories.items() if v["success"].iloc[0]}
    failed_keys  = {k for k, v in histories.items() if not v["success"].iloc[0]}

    cols = ["best_PESQ", "best_SET_OVERLAP"]

    # Check which cols exist
    sample = next(iter(histories.values())) if histories else pd.DataFrame()
    cols = [c for c in cols if c in sample.columns]
    if not cols:
        print("[Skip] fitness_history.csv missing expected columns.")
        return

    def make_plot(keys, title, filename):
        hist_list = [histories[k] for k in keys if k in histories]
        if not hist_list:
            print(f"[Skip] No data for {filename}.")
            return
        curve = _build_curve(hist_list, cols)

        fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5), squeeze=False)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        labels = {"best_PESQ": "Avg Best PESQ", "best_SET_OVERLAP": "Avg Best SET_OVERLAP"}
        colors = {"best_PESQ": COLOR_PESQ, "best_SET_OVERLAP": COLOR_SO}
        thresholds = {"best_PESQ": PESQ_THRESHOLD, "best_SET_OVERLAP": SET_OVERLAP_THRESHOLD}

        for ax, col in zip(axes[0], cols):
            _draw_curve(ax, curve, col, colors[col], labels[col])
            ax.axhline(thresholds[col], color="black", linestyle=":", linewidth=1.5, alpha=0.5,
                       label=f"Threshold ({thresholds[col]})")
            ax.set_xlabel("Generation", fontsize=11)
            ax.set_ylabel("Fitness Score (lower = better)", fontsize=11)
            ax.set_title(labels[col], fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate n contributing runs per generation as faded line
        n_col = cols[0]
        n_per_gen = pd.concat([histories[k] for k in keys if k in histories]).groupby("generation")[n_col].count()
        ax2 = axes[0][-1].twinx()
        ax2.plot(n_per_gen.index, n_per_gen.values, color="gray", linewidth=1,
                 linestyle="--", alpha=0.5, label="N runs contributing")
        ax2.set_ylabel("Runs contributing", fontsize=9, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches="tight")
        print(f"[Saved] {filename}")
        plt.close()

    make_plot(set(histories.keys()), "Average Convergence Curve (All Runs)",
              "convergence_curve_all.png")
    make_plot(failed_keys, "Average Convergence Curve (Failed Runs Only)",
              "convergence_curve_failed.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 8 — Example run composite
# ─────────────────────────────────────────────────────────────────────────────

def plot_example_run(df: pd.DataFrame, output_dir: str, example_run: str = None):
    successful = df[df["success"] & df["generation_found"].notna()]
    if successful.empty:
        print("[Skip] No successful runs — skipping example run.")
        return

    if example_run:
        try:
            sid, rid = map(int, example_run.split(":"))
            row = df[(df["sentence_id"] == sid) & (df["run_id"] == rid)].iloc[0]
        except Exception:
            print(f"[Warn] --example_run '{example_run}' not found, falling back to median.")
            row = None
    else:
        row = None

    if row is None:
        median_gen = successful["generation_found"].median()
        idx = (successful["generation_found"] - median_gen).abs().idxmin()
        row = successful.loc[idx]

    run_dir = Path(row["json_path"]).parent
    print(f"[Example] Sentence {row['sentence_id']}, Run {row['run_id']} "
          f"(generation_found={int(row['generation_found'])})")

    graph_files = [
        ("pareto_evolution.png",      "Pareto Front Evolution"),
        ("hypervolume_convergence.png","Hypervolume Convergence"),
        ("mean_fitness_stack.png",    "Mean Fitness per Generation"),
        ("difference_spectrogram.png", "Difference Spectrogram"),
    ]

    images = [(title, run_dir / fname) for fname, title in graph_files]
    images = [(title, p) for title, p in images if p.exists()]

    if not images:
        print("[Skip] No per-run graphs found in example run directory.")
        return

    n = len(images)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = np.array(axes).flatten()

    for ax, (title, path) in zip(axes, images):
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(
        f"Example Run — Sentence {int(row['sentence_id'])}, Run {int(row['run_id'])}\n"
        f"Converged at Generation {int(row['generation_found'])}  |  "
        f"PESQ={row['pesq']:.3f}  SET_OVERLAP={row['set_overlap']:.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "example_run.png"), dpi=200, bbox_inches="tight")
    print("[Saved] example_run.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 9 — Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_table(df: pd.DataFrame, output_dir: str, n_rows: int = 12):
    successful = df[df["success"]].copy()
    if successful.empty:
        print("[Skip] No successful runs — skipping comparison table.")
        return

    # Pick a spread across different sentences
    sample = (successful
              .sort_values(["sentence_id", "run_id"])
              .drop_duplicates(subset="sentence_id")
              .head(n_rows))

    def wrap(text, width=45):
        return "\n".join(textwrap.wrap(str(text), width)) if text else ""

    cell_data = []
    for _, row in sample.iterrows():
        cell_data.append([
            str(int(row["sentence_id"])),
            wrap(row["ground_truth_text"]),
            wrap(row["asr_transcription"]),
            f"{row['pesq']:.3f}",
            f"{row['set_overlap']:.3f}",
        ])

    col_labels  = ["ID", "Ground Truth", "ASR Transcription", "PESQ", "SET_OVERLAP"]
    col_widths  = [0.05, 0.35, 0.35, 0.12, 0.13]

    row_height  = 0.12
    fig_height  = max(4, len(cell_data) * row_height * 6 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.5)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(cell_data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax.set_title("Successful Attacks — Ground Truth vs ASR Transcription",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_table.png"), dpi=200, bbox_inches="tight")
    print("[Saved] comparison_table.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 10 — All semantic distances per run
# ─────────────────────────────────────────────────────────────────────────────

def plot_semantic_distance_all(df: pd.DataFrame, output_dir: str):
    if "semantic_similarity" not in df.columns or df["semantic_similarity"].isna().all():
        print("[Skip] No semantic_similarity data available.")
        return

    data = df.dropna(subset=["semantic_similarity"]).copy()
    data = data.sort_values(["sentence_id", "run_id"])

    # Use a sequential index as x so runs are evenly spaced
    data = data.reset_index(drop=True)
    x = data.index.values

    success_mask = data["success"].astype(bool)

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.12), 5))

    ax.scatter(x[~success_mask], data.loc[~success_mask, "semantic_similarity"],
               color=COLOR_FAIL,    s=30, alpha=0.75, edgecolors="none",
               label=f"Failure ({(~success_mask).sum()})", zorder=3)
    ax.scatter(x[success_mask],  data.loc[success_mask,  "semantic_similarity"],
               color=COLOR_SUCCESS, s=30, alpha=0.75, edgecolors="none",
               label=f"Success ({success_mask.sum()})", zorder=3)

    # Mean and median reference lines
    mean_val   = data["semantic_similarity"].mean()
    median_val = data["semantic_similarity"].median()
    ax.axhline(mean_val,   color=COLOR_PESQ,   linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.3f}")
    ax.axhline(median_val, color=COLOR_PARTIAL, linestyle="--", linewidth=1.5,
               label=f"Median: {median_val:.3f}")

    # Shade sentence boundaries to aid readability
    sentence_ids = data["sentence_id"].values
    boundaries   = np.where(np.diff(sentence_ids) != 0)[0] + 1
    for i, b in enumerate(boundaries):
        if i % 2 == 0:
            left  = boundaries[i - 1] if i > 0 else 0
            right = b
            ax.axvspan(left, right, color="gray", alpha=0.06, zorder=1)

    ax.set_xlabel("Run (ordered by sentence_id, run_id)", fontsize=11)
    ax.set_ylabel("Semantic Similarity (↓ = attack more effective)", fontsize=11)
    ax.set_title("Sentence Embedding Cosine Similarity — All Runs", fontsize=14, fontweight="bold")
    ax.set_xlim(-1, len(data))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "semantic_similarity_all.png"), dpi=200, bbox_inches="tight")
    print("[Saved] semantic_similarity_all.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Graph 11 — IQR box plots for continuous metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_iqr(df: pd.DataFrame, output_dir: str):
    metrics = [
        ("semantic_similarity", "Semantic Similarity (↓ = attack effective)", None),
        ("set_overlap",       "SET_OVERLAP (↓ better)",       SET_OVERLAP_THRESHOLD),
        ("pesq",              "PESQ (↓ better)",              PESQ_THRESHOLD),
    ]

    # Drop metrics not present in this dataset
    metrics = [(col, label, thr) for col, label, thr in metrics if col in df.columns and df[col].notna().any()]
    if not metrics:
        print("[Skip] No numeric metrics available for IQR plot.")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    groups       = [df[df["success"]], df[~df["success"]]]
    group_labels = ["Success", "Failure"]
    group_colors = [COLOR_SUCCESS, COLOR_FAIL]

    for ax, (col, label, threshold) in zip(axes, metrics):
        data        = [g[col].dropna().values for g in groups]
        positions   = [1, 2]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, alpha=0.5, linestyle="none"),
        )

        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for flier, color in zip(bp["fliers"], group_colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        # Annotate IQR values
        for pos, grp_data, color in zip(positions, data, group_colors):
            if len(grp_data) == 0:
                continue
            q1, med, q3 = np.percentile(grp_data, [25, 50, 75])
            iqr = q3 - q1
            ax.text(pos, q3 + 0.02, f"IQR={iqr:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=color, fontweight="bold")

        if threshold is not None:
            ax.axhline(threshold, color="black", linestyle=":", linewidth=1.5,
                       alpha=0.6, label=f"Threshold ({threshold})")
            ax.legend(fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{l}\n(n={len(g[col].dropna())})" for l, g in zip(group_labels, groups)],
            fontsize=10,
        )
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.suptitle("Metric Distributions with IQR — Success vs Failure",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iqr_boxplots.png"), dpi=200, bbox_inches="tight")
    print("[Saved] iqr_boxplots.png")
    plt.close()


def plot_utmos(df: pd.DataFrame, output_dir: str):
    """Graph 12: UTMOS MOS scores — GT vs adversarial, grouped by success/failure."""
    sub = df.dropna(subset=["utmos_best", "utmos_gt"])
    if sub.empty:
        print("[Skip] plot_utmos: no utmos data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("UTMOS Naturalness Scores (1=worst, 5=best)", fontsize=13, fontweight="bold")

    # ── Left: box plot GT vs adversarial by success ──────────────────────────
    ax = axes[0]
    groups = {
        ("GT",       True):  sub.loc[sub["success"],  "utmos_gt"].dropna(),
        ("GT",       False): sub.loc[~sub["success"], "utmos_gt"].dropna(),
        ("Adversarial", True):  sub.loc[sub["success"],  "utmos_best"].dropna(),
        ("Adversarial", False): sub.loc[~sub["success"], "utmos_best"].dropna(),
    }
    positions = [1, 2, 4, 5]
    colors    = [COLOR_SUCCESS, COLOR_FAIL, COLOR_SUCCESS, COLOR_FAIL]
    labels    = ["GT\n(success)", "GT\n(fail)", "Adv.\n(success)", "Adv.\n(fail)"]
    bp = ax.boxplot(
        [g.values for g in groups.values()],
        positions=positions,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MOS (1–5)")
    ax.set_ylim(1, 5)
    ax.set_title("Distribution by success/failure")
    ax.axvline(3, color="gray", linestyle="--", linewidth=0.8)

    # ── Right: scatter GT vs adversarial ─────────────────────────────────────
    ax = axes[1]
    for success, color, label in [(True, COLOR_SUCCESS, "Success"), (False, COLOR_FAIL, "Failure")]:
        mask = sub["success"] == success
        ax.scatter(
            sub.loc[mask, "utmos_gt"],
            sub.loc[mask, "utmos_best"],
            c=color, alpha=0.6, s=30, label=label, zorder=3,
        )
    lo, hi = 1.0, 5.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x (no change)")
    ax.set_xlabel("GT UTMOS")
    ax.set_ylabel("Adversarial UTMOS")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("GT vs. adversarial naturalness")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "utmos_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] utmos_scores.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze adversarial TTS results")
    parser.add_argument("results_dir",
                        help="Directory containing sentence_XXX/run_N/run_summary.json files")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots (default: results_dir/analysis/)")
    parser.add_argument("--example_run", default=None,
                        help="Specific run to use for graph 8, format: sentence_id:run_id (e.g. 51:0)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)

    if df.empty:
        print("[!] No run_summary.json files found.")
        return

    print(f"[*] Loaded {len(df)} runs across {df['sentence_id'].nunique()} sentences.")

    print_summary(df)

    plot_convergence_bar(df, output_dir)
    plot_cumulative_success(df, output_dir)
    plot_success_pie(df, output_dir)
    plot_sentence_outcome_pie(df, output_dir)
    plot_failed_proximity(df, output_dir)

    print("[*] Loading fitness histories...")
    histories = load_fitness_histories(args.results_dir, df)
    print(f"[*] Loaded {len(histories)} fitness history files.")
    plot_convergence_curves(histories, df, output_dir)

    plot_example_run(df, output_dir, args.example_run)
    plot_comparison_table(df, output_dir)
    plot_semantic_distance_all(df, output_dir)
    plot_iqr(df, output_dir)
    plot_utmos(df, output_dir)

    csv_path = os.path.join(output_dir, "all_results.csv")
    df.drop(columns=["json_path"]).to_csv(csv_path, index=False)
    print(f"[Saved] all_results.csv")


if __name__ == "__main__":
    main()

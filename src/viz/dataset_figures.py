"""
Outputs (under results/figures/dataset/):
  - tier_split_counts.png           bar chart of dataset sizes
  - tier_qtype_heatmap.png          heatmap of question-type counts by tier
  - length_boxplots.png             question/answer/context length box plots
  - noboolq_label_balance.png       Yes/No class balance per split
  - source_distribution.png         NorQuAD wiki vs news per split, per tier
  - answer_len_hist.png             answer length histogram by tier
  - context_len_hist.png            context length histogram by tier+source

Write a JSON summary results/metrics/dataset_summary.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import FIGS, METRICS
from src.data.loader import gold_answer, load_tier, qtype, wstoks

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
OUT = FIGS / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)


def _frame() -> pd.DataFrame:
    rows = []
    for tier in ("yes_no", "short_factual", "long_span"):
        tr, va, te = load_tier(tier)
        for split, data in (("train", tr), ("val", va), ("test", te)):
            for r in data:
                gold = gold_answer(r)
                rows.append({
                    "tier": tier,
                    "split": split,
                    "source": r.get("source", "no_boolq"),
                    "qlen": wstoks(r["question"]),
                    "alen": wstoks(gold),
                    "clen": wstoks(r["context"]),
                    "qtype": qtype(r["question"]),
                    "label": r.get("label"),
                })
    df = pd.DataFrame(rows)
    return df


def fig_split_counts(df: pd.DataFrame) -> None:
    g = df.groupby(["tier", "split"]).size().unstack(fill_value=0)
    g = g[["train", "val", "test"]].loc[["yes_no", "short_factual", "long_span"]]
    ax = g.plot(kind="bar", figsize=(7.5, 4.2), edgecolor="black", linewidth=0.4)
    ax.set_ylabel("# QA pairs")
    ax.set_xlabel("Tier")
    ax.set_title("Dataset sizes by tier and split")
    ax.legend(title="Split")
    plt.xticks(rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(OUT / "tier_split_counts.png", dpi=160)
    plt.savefig(OUT / "tier_split_counts.pdf")
    plt.close()


def fig_qtype_heatmap(df: pd.DataFrame) -> None:
    order_q = ["yes_no", "what", "who", "when", "where", "which",
              "how", "how_many", "how_much", "other"]
    pivot = (
        df[df["split"] == "train"]
          .groupby(["tier", "qtype"]).size()
          .unstack(fill_value=0)
          .reindex(index=["yes_no", "short_factual", "long_span"], columns=order_q, fill_value=0)
    )
    plt.figure(figsize=(9, 3.4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="rocket_r",
                cbar_kws={"label": "# train examples"})
    plt.title("Question-type distribution by tier (train split)")
    plt.xlabel("Question type (inferred from leading word)")
    plt.ylabel("Tier")
    plt.tight_layout()
    plt.savefig(OUT / "tier_qtype_heatmap.png", dpi=160)
    plt.savefig(OUT / "tier_qtype_heatmap.pdf")
    plt.close()


def fig_length_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    order = ["yes_no", "short_factual", "long_span"]
    for ax, col, title, ymax in zip(
        axes,
        ["qlen", "alen", "clen"],
        ["Question length (tokens)", "Answer length (tokens)", "Context length (tokens)"],
        [40, 30, 600],
    ):
        sns.boxplot(data=df, x="tier", y=col, order=order, ax=ax,
                    showfliers=False, hue="tier", legend=False, palette="Set2")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, ymax)
    plt.suptitle("Length statistics across tiers (whiskers = 1.5 IQR, outliers hidden)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "length_boxplots.png", dpi=160)
    plt.savefig(OUT / "length_boxplots.pdf")
    plt.close()


def fig_noboolq_label_balance(df: pd.DataFrame) -> None:
    sub = df[df["tier"] == "yes_no"].copy()
    sub["answer"] = sub["label"].map({True: "Ja (Yes)", False: "Nei (No)"})
    g = sub.groupby(["split", "answer"]).size().unstack(fill_value=0)
    g = g.loc[["train", "val", "test"]]
    ax = g.plot(kind="bar", stacked=False, figsize=(6.5, 4),
                color=["#4c9b6d", "#c4615a"], edgecolor="black", linewidth=0.4)
    ax.set_title("NO-BoolQ class balance per split")
    ax.set_xlabel("Split")
    ax.set_ylabel("# examples")
    ax.legend(title="Label")
    plt.xticks(rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fontsize=9, padding=2)
    plt.tight_layout()
    plt.savefig(OUT / "noboolq_label_balance.png", dpi=160)
    plt.savefig(OUT / "noboolq_label_balance.pdf")
    plt.close()


def fig_source_distribution(df: pd.DataFrame) -> None:
    nq = df[df["tier"].isin(["short_factual", "long_span"])].copy()
    g = nq.groupby(["tier", "split", "source"]).size().unstack("source", fill_value=0)
    g = g.reindex(["news", "wiki", "unknown"], axis=1, fill_value=0)
    g = g.reindex(
        index=[(t, s) for t in ["short_factual", "long_span"]
                       for s in ["train", "val", "test"]],
        fill_value=0,
    )
    ax = g.plot(kind="bar", stacked=True, figsize=(8, 4),
                color=["#5b8def", "#f4a259", "#bbbbbb"],
                edgecolor="black", linewidth=0.4)
    ax.set_title("NorQuAD source distribution (news vs Wikipedia)")
    ax.set_ylabel("# examples")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"{a}\n{b}" for a, b in g.index], rotation=0,
    )
    ax.legend(title="Source")
    plt.tight_layout()
    plt.savefig(OUT / "source_distribution.png", dpi=160)
    plt.savefig(OUT / "source_distribution.pdf")
    plt.close()


def fig_answer_len_hist(df: pd.DataFrame) -> None:
    sub = df[df["tier"].isin(["short_factual", "long_span"])]
    plt.figure(figsize=(7.5, 4))
    for tier, color in (("short_factual", "#3a86ff"), ("long_span", "#fb5607")):
        d = sub[(sub["tier"] == tier) & (sub["split"] == "train")]["alen"]
        plt.hist(d, bins=np.arange(0, 40, 1), alpha=0.6, label=tier, color=color, edgecolor="black", linewidth=0.3)
    plt.axvline(5.5, color="black", linestyle="--", linewidth=1, label=r"tier cutoff (5 tokens)")
    plt.title("NorQuAD answer-length distribution (train)")
    plt.xlabel("Answer length (whitespace tokens)")
    plt.ylabel("# examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "answer_len_hist.png", dpi=160)
    plt.savefig(OUT / "answer_len_hist.pdf")
    plt.close()


def fig_context_len_hist(df: pd.DataFrame) -> None:
    sub = df[df["tier"].isin(["yes_no", "short_factual", "long_span"])
             & (df["split"] == "train")]
    plt.figure(figsize=(7.5, 4))
    for tier, color in (
        ("yes_no", "#2a9d8f"),
        ("short_factual", "#3a86ff"),
        ("long_span", "#fb5607"),
    ):
        d = sub[sub["tier"] == tier]["clen"]
        plt.hist(d.clip(0, 800), bins=40, alpha=0.55, label=tier, color=color,
                 edgecolor="black", linewidth=0.3)
    plt.title("Context-length distribution by tier (train, clipped at 800 tokens)")
    plt.xlabel("Context length (whitespace tokens)")
    plt.ylabel("# examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "context_len_hist.png", dpi=160)
    plt.savefig(OUT / "context_len_hist.pdf")
    plt.close()


def write_summary(df: pd.DataFrame) -> None:
    summary = {}
    for tier in ("yes_no", "short_factual", "long_span"):
        sub = df[df["tier"] == tier]
        summary[tier] = {}
        for split in ("train", "val", "test"):
            s = sub[sub["split"] == split]
            row = {
                "n": int(len(s)),
                "qlen_mean": float(s["qlen"].mean()),
                "qlen_median": float(s["qlen"].median()),
                "alen_mean": float(s["alen"].mean()),
                "alen_median": float(s["alen"].median()),
                "clen_mean": float(s["clen"].mean()),
                "clen_median": float(s["clen"].median()),
                "sources": {k: int(v) for k, v in Counter(s["source"]).items()},
                "qtypes": {k: int(v) for k, v in Counter(s["qtype"]).items()},
            }
            if tier == "yes_no":
                row["label_pos"] = int((s["label"] == True).sum())
                row["label_neg"] = int((s["label"] == False).sum())
            summary[tier][split] = row
    (METRICS / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


def main() -> None:
    df = _frame()
    print(f"[viz] tidy frame: {len(df)} rows")
    fig_split_counts(df)
    fig_qtype_heatmap(df)
    fig_length_boxplots(df)
    fig_noboolq_label_balance(df)
    fig_source_distribution(df)
    fig_answer_len_hist(df)
    fig_context_len_hist(df)
    write_summary(df)
    print(f"[viz] wrote figures -> {OUT}")
    print(f"[viz] wrote summary -> {METRICS / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()

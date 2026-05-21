# Build the results figures from results/metrics/*.json and results/predictions/*.jsonl.

"""
Outputs (under results/figures/results/):
  - model_comparison_<tier>.png  bar chart with 95% bootstrap CIs
  - confusion_yes_no.png         confusion matrices (one panel per model)
  - per_qtype_f1_heatmap.png     model x question-type F1
  - training_curves__*.png       loss / val metric curves from logs/*.json
  - summary_table.csv            tidy headline table
  """

# The figure builder is flexible, if a metric file is missing it skips that model.

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import FIGS, LOGS, METRICS, PREDS
from src.data.loader import qtype, wstoks

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
OUT = FIGS / "results"
OUT.mkdir(parents=True, exist_ok=True)


TIER_FROM_TASK = {
    "boolq": "yes_no",
    "squad_short": "short_factual",
    "squad_long": "long_span",
    "squad_all": "extractive_all",
}


def parse_pred_filename(path: Path) -> dict | None:
    name = path.name
    if not name.endswith("__test.jsonl"):
        return None
    stem = name[: -len("__test.jsonl")]
    parts = stem.split("__")
    info = {"file": str(path), "stem": stem}
    if parts[0] == "bidaf" and len(parts) == 2:
        info["model"] = "BiDAF"
        info["task"] = parts[1]
        info["tier"] = TIER_FROM_TASK.get(info["task"], info["task"])
        info["family"] = "baseline"
    elif parts[0] == "encoder" and len(parts) == 3:
        info["model"] = parts[1].replace("_", "/")
        info["task"] = parts[2]
        info["tier"] = TIER_FROM_TASK.get(info["task"], info["task"])
        info["family"] = "encoder"
    elif parts[0] == "zs" and len(parts) == 4:
        info["model"] = parts[1]
        info["tier"] = parts[2]
        info["cfg"] = parts[3]
        info["family"] = "generative"
    elif parts[0] == "zs" and len(parts) == 3:
        info["model"] = parts[1]
        info["tier"] = parts[2]
        info["cfg"] = "free"
        info["family"] = "generative"
    elif parts[0] == "baseline" and len(parts) == 3:
        info["model"] = parts[1]
        info["tier"] = parts[2]
        info["family"] = "baseline_lex"
    else:
        return None
    return info


def discover_runs() -> pd.DataFrame:
    rows = []
    for p in sorted(PREDS.glob("*__test.jsonl")):
        info = parse_pred_filename(p)
        if info is None:
            continue
        mfile = METRICS / f"{p.stem}.json"
        info["metrics_file"] = str(mfile) if mfile.exists() else None
        rows.append(info)
    df = pd.DataFrame(rows)
    print(f"[viz/results] discovered {len(df)} prediction files")
    return df


def load_metric(metrics_file: str | None) -> dict | None:
    if metrics_file is None:
        return None
    try:
        return json.loads(Path(metrics_file).read_text())
    except Exception:
        return None


# per-tier model comparison bars

PRIMARY = {
    "yes_no": ("macro_f1", "accuracy"),
    "short_factual": ("f1", "em"),
    "long_span": ("f1", "rouge_l"),
    "extractive_all": ("f1", "em"),
}


def fig_model_comparison(runs: pd.DataFrame) -> None:
    for tier in ("yes_no", "short_factual", "long_span"):
        sub = runs[runs["tier"] == tier].copy()
        if sub.empty:
            continue
        prim, sec = PRIMARY[tier]
        rows = []
        for _, r in sub.iterrows():
            m = load_metric(r.get("metrics_file"))
            if m is None:
                continue
            rec = {"model": _pretty_model(r), "family": r["family"]}
            rec[prim] = m.get(prim, np.nan)
            rec[sec] = m.get(sec, np.nan)
            if "abstain_rate" in m:
                rec["abstain_rate"] = m["abstain_rate"]
            ci_key = {"macro_f1": "macro_f1_ci95",
                      "f1": "f1_ci95"}.get(prim)
            if ci_key and ci_key in m:
                rec["lo"] = m[ci_key][0]; rec["hi"] = m[ci_key][1]
            elif prim == "macro_f1" and "accuracy_ci95" in m:
                w = (m["accuracy_ci95"][1] - m["accuracy_ci95"][0]) / 2
                rec["lo"] = rec[prim] - w; rec["hi"] = rec[prim] + w
            else:
                rec["lo"] = rec["hi"] = rec[prim]
            rows.append(rec)
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(prim, ascending=True)
        n = len(df)
        plt.figure(figsize=(8.5, max(3, n * 0.5)))
        ypos = np.arange(n)
        family_color = {"baseline_lex": "#bbbbbb", "baseline": "#aaaaaa",
                        "encoder": "#3a86ff", "generative": "#fb5607"}
        colors = [family_color.get(f, "#555555") for f in df["family"]]
        err_lo = np.clip(df[prim] - df["lo"], 0, None)
        err_hi = np.clip(df["hi"] - df[prim], 0, None)
        plt.barh(ypos, df[prim], xerr=[err_lo, err_hi], color=colors,
                 edgecolor="black", linewidth=0.4)
        plt.yticks(ypos, df["model"])
        plt.xlabel(f"{prim} (95% bootstrap CI)")
        plt.title(f"{tier}: model comparison")
        plt.xlim(0, 1)
        for y, row in enumerate(df.itertuples()):
            pv = getattr(row, prim)
            sv = getattr(row, sec)
            label = f"{prim}={pv:.3f}\n{sec}={sv:.3f}"
            ab = getattr(row, "abstain_rate", None)
            if ab is not None and ab > 0.01:
                label += f"\nabstain={ab:.2f}"
            plt.text(min(pv + 0.02, 0.97), y, label, fontsize=7.5, va="center")
        plt.tight_layout()
        plt.savefig(OUT / f"model_comparison_{tier}.png", dpi=160)
        plt.savefig(OUT / f"model_comparison_{tier}.pdf")
        plt.close()


def _cfg_suffix(cfg: str | None) -> str:
    if cfg is None or cfg == "free":
        return "zero-shot"
    if cfg == "logit_bias":
        return "logit-bias"
    m = re.fullmatch(r"fewshot(\d+)", cfg or "")
    if m:
        return f"{m.group(1)}-shot"
    m = re.fullmatch(r"logit_bias_fewshot(\d+)", cfg or "")
    if m:
        return f"logit-bias + {m.group(1)}-shot"
    return cfg


def _pretty_model(r) -> str:
    fam = r["family"]; m = r["model"]
    if fam == "baseline":
        return "BiDAF"
    if fam == "baseline_lex":
        return f"{m} (baseline)"
    if fam == "encoder":
        s = m.replace("NbAiLab/", "").replace("ltg/", "")
        return s
    cfg = r.get("cfg", "free")
    if not isinstance(cfg, str):
        cfg = "free"
    suffix = _cfg_suffix(cfg)
    if "7b_lora" in m:
        return f"NorMistral-7B-warm + LoRA ({suffix})"
    if "11b_thinking" in m and "closed" in m:
        return f"NorMistral-11B-thinking (closed-book, {suffix})"
    if "11b_thinking" in m:
        return f"NorMistral-11B-thinking ({suffix})"
    if "7b_warm" in m:
        return f"NorMistral-7B-warm ({suffix})"
    return f"{m} ({suffix})"


# confusion matrices (yes/no)

def fig_confusion_yesno(runs: pd.DataFrame) -> None:
    sub = runs[runs["tier"] == "yes_no"].copy()
    if sub.empty:
        return
    n = len(sub)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3 * rows), squeeze=False)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax = axes[i // cols, i % cols]
        m = load_metric(r.get("metrics_file"))
        if m is None or "confusion" not in m:
            ax.text(0.5, 0.5, "missing", ha="center"); ax.axis("off"); continue
        c = m["confusion"]
        ab_ja = c.get("abstain_ja", 0)
        ab_nei = c.get("abstain_nei", 0)
        has_abstain = (ab_ja + ab_nei) > 0
        if has_abstain:
            mat = np.array([
                [c.get("tp_ja", 0), c.get("fn_ja", 0), ab_ja],
                [c.get("fp_ja", 0), c.get("tn_nei", 0), ab_nei],
            ], dtype=float)
            xticklabels = ["pred Ja", "pred Nei", "abstain"]
        else:
            mat = np.array([
                [c.get("tp_ja", 0), c.get("fn_ja", 0)],
                [c.get("fp_ja", 0), c.get("tn_nei", 0)],
            ], dtype=float)
            xticklabels = ["pred Ja", "pred Nei"]
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                    xticklabels=xticklabels,
                    yticklabels=["gold Ja", "gold Nei"], cbar=False)
        ax.set_title(_pretty_model(r), fontsize=9)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    plt.suptitle("NO-BoolQ confusion matrices", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_yes_no.png", dpi=160)
    plt.savefig(OUT / "confusion_yes_no.pdf")
    plt.close()


# per-question-type F1 heatmap

def fig_per_qtype_f1(runs: pd.DataFrame) -> None:
    sub = runs[runs["tier"].isin(("short_factual", "long_span"))].copy()
    if sub.empty:
        return
    qtypes_order = ["when", "where", "who", "what", "which",
                    "how", "how_many", "other"]
    table = defaultdict(dict)
    for _, r in sub.iterrows():
        rows = [json.loads(l) for l in open(r["file"]) if l.strip()]
        per_q = defaultdict(list)
        for ex in rows:
            qt = qtype(ex.get("question", ""))
            from src.eval.score import token_f1
            per_q[qt].append(token_f1(ex.get("pred", ""), ex.get("gold", "")))
        label = f"{_pretty_model(r)} ({r['tier']})"
        for q in qtypes_order:
            if per_q[q]:
                table[label][q] = float(np.mean(per_q[q]))
    if not table:
        return
    df = pd.DataFrame(table).T.reindex(columns=qtypes_order)
    plt.figure(figsize=(1 + 0.8 * len(qtypes_order), 0.45 * len(df) + 1.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "token F1"})
    plt.title("Per-question-type token-F1 (NorQuAD test)")
    plt.tight_layout()
    plt.savefig(OUT / "per_qtype_f1_heatmap.png", dpi=160)
    plt.savefig(OUT / "per_qtype_f1_heatmap.pdf")
    plt.close()


# training curves

def fig_training_curves() -> None:
    for log_path in sorted(LOGS.glob("*.json")):
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            continue
        if not log.get("step") or not log.get("epoch"):
            continue
        steps = [s["step"] for s in log["step"]]
        losses = [s["loss"] for s in log["step"]]
        epoch_x = [e.get("step_end", 0) for e in log["epoch"]]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(steps, losses, color="#444444", linewidth=0.9, label="train loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss", color="#444444")
        ax2 = ax1.twinx()
        for k in ("val_f1", "val_em", "val_macroF1", "val_acc"):
            ys = [e.get(k) for e in log["epoch"]]
            if any(y is not None for y in ys):
                ax2.plot(epoch_x, ys, "o-", label=k, linewidth=1.4)
        ax2.set_ylabel("val metric")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right")
        plt.title(log_path.stem)
        plt.tight_layout()
        plt.savefig(OUT / f"training_curves__{log_path.stem}.png", dpi=140)
        plt.close()


# summary table

def write_summary_table(runs: pd.DataFrame) -> None:
    rows = []
    for _, r in runs.iterrows():
        m = load_metric(r.get("metrics_file"))
        if m is None:
            continue
        row = {
            "tier": r["tier"],
            "family": r["family"],
            "model": _pretty_model(r),
            "stem": r["stem"],
            "n": m.get("n"),
        }
        for k in ("accuracy", "accuracy_committed", "macro_f1", "kappa",
                  "abstain_rate", "em", "f1", "rouge_l"):
            if k in m:
                row[k] = m[k]
        if m.get("bertscore_rescaled") and "bertscore_f1" in m:
            row["bertscore_f1_rescaled"] = m["bertscore_f1"]
        for k in ("accuracy_ci95", "macro_f1_ci95", "f1_ci95", "em_ci95",
                  "rouge_ci95"):
            if k in m:
                row[k] = f"{m[k][0]:.4f}/{m[k][1]:.4f}"
        if m.get("bertscore_rescaled") and "bertscore_ci95" in m:
            row["bertscore_ci95"] = f"{m['bertscore_ci95'][0]:.4f}/{m['bertscore_ci95'][1]:.4f}"
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["tier", "family", "model"])
    df.to_csv(OUT / "summary_table.csv", index=False)
    print(f"[viz/results] wrote {OUT / 'summary_table.csv'} ({len(df)} rows)")


def main():
    runs = discover_runs()
    fig_model_comparison(runs)
    fig_confusion_yesno(runs)
    fig_per_qtype_f1(runs)
    fig_training_curves()
    write_summary_table(runs)
    print(f"[viz/results] figures -> {OUT}")


if __name__ == "__main__":
    main()

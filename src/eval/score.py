from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import METRICS


# normalization & lexical metrics

def normalize(s: str) -> str:
    # lower, strip punctuation, collapse whitespace
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    nsame = sum(common.values())
    if nsame == 0:
        return 0.0
    pr = nsame / len(p)
    rc = nsame / len(g)
    return 2 * pr * rc / (pr + rc)


def rouge_l(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    # LCS DP
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    pr = lcs / m
    rc = lcs / n
    return 2 * pr * rc / (pr + rc)


# yes/no metrics

# Words we accept as an explicit commitment to Ja / Nei. Anything outside these sets is treated as a non-commit (abstention).
JA_TOKENS = {"ja", "sant", "stemmer", "yes", "y", "true", "1"}
NEI_TOKENS = {"nei", "usant", "feil", "no", "n", "false", "0"}
DISJUNCTIONS = {"eller", "or"}


def parse_yesno(s: str) -> str | None:
    n = normalize(s)
    if not n:
        return None
    toks = n.split()
    has_ja = any(t in JA_TOKENS for t in toks)
    has_nei = any(t in NEI_TOKENS for t in toks)
    has_disj = any(t in DISJUNCTIONS for t in toks)
    # Strings that mention both classes, or use the disjunction "eller",
    # are treated as non-commits even if they happen to start with "Ja".
    if has_disj or (has_ja and has_nei):
        return None
    if toks[0] in JA_TOKENS:
        return "Ja"
    if toks[0] in NEI_TOKENS:
        return "Nei"
    return None


def score_yesno(rows: list[dict]) -> dict:
    n = len(rows)
    tp = fp = tn = fn = 0
    abstain_ja = abstain_nei = 0  # abstentions split by gold class
    for r in rows:
        g = parse_yesno(r["gold"])
        p = parse_yesno(r["pred"])
        if g is None:
            # gold itself is unparseable -> skip; should not happen for NO-BoolQ
            continue
        if p is None:
            if g == "Ja":
                abstain_ja += 1
            else:
                abstain_nei += 1
            continue
        if g == "Ja" and p == "Ja":
            tp += 1
        elif g == "Nei" and p == "Nei":
            tn += 1
        elif g == "Ja" and p == "Nei":
            fn += 1
        else:
            fp += 1
    abstain = abstain_ja + abstain_nei
    committed = tp + tn + fp + fn
    acc = (tp + tn) / max(n, 1)
    acc_committed = (tp + tn) / max(committed, 1) if committed else 0.0
    abstain_rate = abstain / max(n, 1)
    f1_ja = _safe_f1(tp, tp + fp, tp + fn + abstain_ja)
    f1_nei = _safe_f1(tn, tn + fn, tn + fp + abstain_nei)
    macro = (f1_ja + f1_nei) / 2
    po = acc
    p_yes_pred = (tp + fp) / max(n, 1)
    g_yes = (tp + fn + abstain_ja) / max(n, 1)
    pe = p_yes_pred * g_yes + (1 - p_yes_pred) * (1 - g_yes)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0
    return {
        "n": n,
        "committed": committed,
        "abstain": abstain,
        "abstain_rate": abstain_rate,
        "accuracy": acc,
        "accuracy_committed": acc_committed,
        "f1_ja": f1_ja,
        "f1_nei": f1_nei,
        "macro_f1": macro,
        "kappa": kappa,
        "confusion": {
            "tp_ja": tp, "fn_ja": fn, "fp_ja": fp, "tn_nei": tn,
            "abstain_ja": abstain_ja, "abstain_nei": abstain_nei,
        },
    }


def _safe_f1(tp, pp, gp):
    if pp == 0 or gp == 0:
        return 0.0
    p = tp / pp; r = tp / gp
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# extractive metrics

def score_extractive(rows: list[dict], compute_bertscore: bool = False) -> dict:
    em_vals = [em(r["pred"], r["gold"]) for r in rows]
    f1_vals = [token_f1(r["pred"], r["gold"]) for r in rows]
    rouge_vals = [rouge_l(r["pred"], r["gold"]) for r in rows]
    out = {
        "n": len(rows),
        "em": float(np.mean(em_vals)) if em_vals else 0.0,
        "f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "rouge_l": float(np.mean(rouge_vals)) if rouge_vals else 0.0,
        "em_per_example": em_vals,
        "f1_per_example": f1_vals,
        "rouge_per_example": rouge_vals,
    }
    if compute_bertscore:
        try:
            from bert_score import score
            P, R, F = score(
                [r["pred"] for r in rows],
                [r["gold"] for r in rows],
                lang="no",
                model_type="xlm-roberta-large",
                num_layers=17,
                rescale_with_baseline=True,
                verbose=False,
            )
            out["bertscore_f1"] = float(F.mean())
            out["bertscore_per_example"] = F.tolist()
            out["bertscore_rescaled"] = True
        except Exception as e:
            try:
                from bert_score import score
                P, R, F = score(
                    [r["pred"] for r in rows],
                    [r["gold"] for r in rows],
                    lang="other",
                    model_type="xlm-roberta-large",
                    num_layers=17,
                    verbose=False,
                )
                out["bertscore_f1"] = float(F.mean())
                out["bertscore_per_example"] = F.tolist()
                out["bertscore_rescaled"] = False
            except Exception as e2:
                out["bertscore_error"] = f"{e}; fallback: {e2}"
    return out


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(arr[idx].mean())
    means = np.array(means)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--bertscore", action="store_true")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.pred, encoding="utf-8") if line.strip()]
    if not rows:
        print("no predictions"); return
    tier = rows[0].get("tier")
    if not tier:
        name = Path(args.pred).stem.lower()
        if "boolq" in name or "yes_no" in name:
            tier = "yes_no"
        elif "squad_long" in name or "long_span" in name:
            tier = "long_span"
        elif "squad_short" in name or "short_factual" in name:
            tier = "short_factual"
        else:
            tier = "short_factual"
        for r in rows:
            r["tier"] = tier
    print(f"[score] {args.pred} tier={tier} n={len(rows)}")

    if tier == "yes_no":
        m = score_yesno(rows)
        rng = np.random.default_rng(42)
        n = len(rows)
        per_ex_acc = []
        macro_samples = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            sample = [rows[i] for i in idx]
            ms = score_yesno(sample)
            macro_samples.append(ms["macro_f1"])
            per_ex_acc.append(ms["accuracy"])
        m["accuracy_ci95"] = [float(np.percentile(per_ex_acc, 2.5)),
                              float(np.percentile(per_ex_acc, 97.5))]
        m["macro_f1_ci95"] = [float(np.percentile(macro_samples, 2.5)),
                              float(np.percentile(macro_samples, 97.5))]
    else:
        m = score_extractive(rows, compute_bertscore=args.bertscore)
        m["em_ci95"] = list(bootstrap_ci(m.pop("em_per_example"), n_boot=args.n_boot))
        m["f1_ci95"] = list(bootstrap_ci(m.pop("f1_per_example"), n_boot=args.n_boot))
        m["rouge_ci95"] = list(bootstrap_ci(m.pop("rouge_per_example"), n_boot=args.n_boot))
        if "bertscore_per_example" in m:
            m["bertscore_ci95"] = list(bootstrap_ci(m.pop("bertscore_per_example"),
                                                    n_boot=args.n_boot))

    out = args.out
    if out is None:
        stem = Path(args.pred).stem
        out = METRICS / f"{stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in m.items() if not k.endswith("_per_example")},
                     indent=2, ensure_ascii=False))
    print(f"[score] -> {out}")


if __name__ == "__main__":
    main()

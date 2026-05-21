# Lexical baselines (no learning) for the three tiers.

"""
  yes_no        majority class on train -> always "Ja" on test
  short_factual first capitalised token-run in the context after the question
                title (a crude proxy for "first named entity")
  long_span     first sentence of the body context
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PREDS
from src.data.loader import load_tier


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# yes/no

def majority_label(train_rows) -> str:
    c = Counter(r["answer_text"] for r in train_rows)
    return c.most_common(1)[0][0]


def baseline_majority(tier: str = "yes_no") -> Path:
    train, _, test = load_tier(tier)
    label = majority_label(train)
    print(f"[baseline/majority] train majority = {label}")
    preds = [
        {"id": r["id"], "tier": tier, "question": r["question"],
         "gold": r["answer_text"], "pred": label}
        for r in test
    ]
    out = PREDS / f"baseline__majority__{tier}__test.jsonl"
    write_jsonl(preds, out)
    print(f"[baseline/majority] wrote {len(preds)} predictions -> {out}")
    return out


# short factual: first capitalised run

# Drop the first paragraph if it looks like a Wikipedia article title (no
# spaces, single capitalised line). The NorQuAD contexts almost always start
# with a title line followed by a blank line and the body.
_TOK_RE = re.compile(r"\S+")
_CAP_RE = re.compile(r"^[A-ZÆØÅ][\wÆØÅæøå.\-]*$")


def _strip_title(ctx: str) -> str:
    # NorQuAD contexts often start with a short title line followed by a
    # newline and then the body. Treat the first line as a title if it has
    # no terminal punctuation and is short (<= ~12 whitespace tokens).
    lines = ctx.splitlines()
    if len(lines) >= 2:
        first = lines[0].strip()
        if first and len(first.split()) <= 12 and first[-1] not in ".!?":
            return "\n".join(lines[1:]).strip()
    return ctx


def first_cap_run(ctx: str, max_tokens: int = 5) -> str:
    body = _strip_title(ctx)
    toks = _TOK_RE.findall(body)
    run = []
    started = False
    for t in toks:
        clean = t.strip(",.;:!?\"'«»()[]")
        if _CAP_RE.match(clean):
            run.append(clean)
            started = True
            if len(run) >= max_tokens:
                break
        elif started:
            break
    return " ".join(run)


def baseline_first_np(tier: str = "short_factual") -> Path:
    _, _, test = load_tier(tier)
    preds = []
    for r in test:
        pred = first_cap_run(r["context"], max_tokens=5)
        preds.append({
            "id": r["id"], "tier": tier,
            "question": r["question"],
            "gold": r["answers"]["text"][0],
            "pred": pred,
        })
    out = PREDS / f"baseline__first_np__{tier}__test.jsonl"
    write_jsonl(preds, out)
    print(f"[baseline/first_np] wrote {len(preds)} predictions -> {out}")
    return out


# long span: first sentence of body

_SENT_END = re.compile(r"[.!?]\s+")


def first_sentence(ctx: str) -> str:
    body = _strip_title(ctx)
    parts = _SENT_END.split(body, maxsplit=1)
    return parts[0].strip()


def baseline_first_sentence(tier: str = "long_span") -> Path:
    _, _, test = load_tier(tier)
    preds = []
    for r in test:
        preds.append({
            "id": r["id"], "tier": tier,
            "question": r["question"],
            "gold": r["answers"]["text"][0],
            "pred": first_sentence(r["context"]),
        })
    out = PREDS / f"baseline__first_sentence__{tier}__test.jsonl"
    write_jsonl(preds, out)
    print(f"[baseline/first_sentence] wrote {len(preds)} predictions -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "majority", "first_np", "first_sentence"])
    args = ap.parse_args()
    if args.which in ("all", "majority"):
        baseline_majority("yes_no")
    if args.which in ("all", "first_np"):
        baseline_first_np("short_factual")
    if args.which in ("all", "first_sentence"):
        baseline_first_sentence("long_span")


if __name__ == "__main__":
    main()

"""
Use:

    from src.data.loader import load_tier
    train, val, test = load_tier("yes_no")         # NO-BoolQ
    train, val, test = load_tier("short_factual")  # NorQuAD answers <=5 tokens
    train, val, test = load_tier("long_span")      # NorQuAD answers > 5 tokens
    train, val, test = load_tier("extractive_all") # all NorQuAD (both tiers)

Returns lists-of-dicts. For "yes_no", every row has
    {id, question, context, label, answer_text, ...}
For NorQuAD tiers every row has
    {id, question, context, answers: {text, answer_start}, ...}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PROC_NOBOOLQ, PROC_NORQUAD

TIER_DIRS = {
    "yes_no": PROC_NOBOOLQ,
    "short_factual": PROC_NORQUAD / "short_factual",
    "long_span": PROC_NORQUAD / "long_span",
    "extractive_all": PROC_NORQUAD,
}

# Norwegian wh-word -> force question type (Bokmål primarily)
NB_QTYPES = [
    ("when",     r"^(når)\b"),
    ("where",    r"^(hvor|kor)\b(?!\s+(mange|mye|stor|liten|gammel|lang|ofte|høy|dyp))"),
    ("who",      r"^(hvem|kven)\b"),
    ("what",     r"^(hva|kva)\b"),
    ("which",    r"^(hvilk|kva for)"),
    ("how_many", r"^(hvor|kor)\s+(mange|mye)\b"),
    ("how_much", r"^(hvor|kor)\s+(mye)\b"),
    ("how",      r"^(hvordan|korleis|hvorfor|kvifor)\b"),
    ("yes_no",   r"^(er|var|har|hadde|gjør|gjorde|kan|kunne|vil|ville|skal|skulle|burde|må|finnes|finn|fins)\b"),
]
_compiled = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in NB_QTYPES]


def qtype(question: str) -> str:
    q = question.strip().lstrip("«\"'(")
    for name, pat in _compiled:
        if pat.search(q):
            return name
    return "other"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tier(tier: str) -> tuple[list[dict], list[dict], list[dict]]:
    base = TIER_DIRS[tier]
    return (
        _read_jsonl(base / "train.jsonl"),
        _read_jsonl(base / "val.jsonl"),
        _read_jsonl(base / "test.jsonl"),
    )


def gold_answer(row: dict) -> str: # Return the first gold answer text for a row from any tier.
    if "answer_text" in row:
        return row["answer_text"]
    return row["answers"]["text"][0]


def wstoks(s: str) -> int:
    return len([t for t in re.split(r"\s+", s.strip()) if t])

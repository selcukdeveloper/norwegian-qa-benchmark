# Process NorQuAD raw JSON into tier-stratified jsonl files.
"""
Tier assignment:
- "short_factual" if gold answer has <= SHORT_MAX_TOKENS whitespace tokens
- "long_span"     otherwise

Each output row is a flat dict suitable for HuggingFace `datasets.load_dataset('json', ...)`:

    {
        "id": str,
        "question": str,
        "context": str,
        "answers": {"text": [str], "answer_start": [int]},
        "tier": "short_factual" | "long_span",
        "source": "wiki" | "news" | "unknown",
        "doc_id": int | None,
        "split": "train" | "val" | "test",
    }

Source ("wiki" vs "news") is inferred by checking which subset file the document
appears in under NorQuAD data/validation/annotator1/{news,wikipedia}/.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    PROC_NORQUAD,
    RAW_NORQUAD,
    ROOT,
    SHORT_MAX_TOKENS,
)

SPLIT_FILES = {
    "train": "training_dataset_flattened.json",
    "val":   "validation_dataset_flattened.json",
    "test":  "test_dataset_flattened.json",
}


def _build_qid_source_map() -> dict[str, str]: # Use evaluation/{news,wiki}/*.json to learn question_id -> source.
    mapping: dict[str, str] = {}
    base = ROOT / "NorQuAD data" / "evaluation"
    for src_dir, label in (("news", "news"), ("wiki", "wiki")):
        for split_file in SPLIT_FILES.values():
            p = base / src_dir / split_file
            if not p.exists():
                continue
            obj = json.loads(p.read_text())
            for doc in obj["data"]:
                for para in doc["paragraphs"]:
                    for qa in para["qas"]:
                        mapping[str(qa["id"])] = label
    return mapping


def _wstoks(s: str) -> int:
    return len([t for t in re.split(r"\s+", s.strip()) if t])


def _normalize_answers(qa: dict) -> dict:
    texts, starts = [], []
    for a in qa.get("answers") or []:
        t = a.get("text") or ""
        s = a.get("answer_start")
        if not t:
            continue
        texts.append(t)
        starts.append(int(s) if isinstance(s, int) else -1)
    return {"text": texts, "answer_start": starts}


def process_split(split: str, src_map: dict[str, str]) -> list[dict]:
    src = RAW_NORQUAD / SPLIT_FILES[split]
    raw = json.loads(src.read_text())
    out: list[dict] = []
    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            ctx = para["context"]
            doc_id = para.get("document_id")
            for qa in para["qas"]:
                ans = _normalize_answers(qa)
                if not ans["text"]:
                    continue
                # primary answer = first gold
                gold = ans["text"][0]
                tier = "short_factual" if _wstoks(gold) <= SHORT_MAX_TOKENS else "long_span"
                source = src_map.get(str(qa["id"]), "unknown")
                out.append({
                    "id": str(qa["id"]),
                    "question": qa["question"],
                    "context": ctx,
                    "answers": ans,
                    "tier": tier,
                    "source": source,
                    "doc_id": doc_id,
                    "split": split,
                })
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    PROC_NORQUAD.mkdir(parents=True, exist_ok=True)
    src_map = _build_qid_source_map()
    print(f"[norquad] inferred source for {len(src_map)} unique question_ids")

    all_rows: dict[str, list[dict]] = {}
    for split in SPLIT_FILES:
        rows = process_split(split, src_map)
        all_rows[split] = rows
        write_jsonl(rows, PROC_NORQUAD / f"{split}.jsonl")
        tier_c = Counter(r["tier"] for r in rows)
        src_c = Counter(r["source"] for r in rows)
        print(f"[norquad] {split}: {len(rows)} qa "
              f"(short={tier_c['short_factual']} long={tier_c['long_span']}) "
              f"sources={dict(src_c)}")

    # per-tier convenience files
    for tier in ("short_factual", "long_span"):
        for split in SPLIT_FILES:
            sub = [r for r in all_rows[split] if r["tier"] == tier]
            write_jsonl(sub, PROC_NORQUAD / tier / f"{split}.jsonl")
        print(f"[norquad] wrote tier '{tier}' splits")

    # summary
    summary = {
        split: {
            "n": len(all_rows[split]),
            "tiers": dict(Counter(r["tier"] for r in all_rows[split])),
            "sources": dict(Counter(r["source"] for r in all_rows[split])),
        } for split in SPLIT_FILES
    }
    (PROC_NORQUAD / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[norquad] done ->", PROC_NORQUAD)


if __name__ == "__main__":
    main()

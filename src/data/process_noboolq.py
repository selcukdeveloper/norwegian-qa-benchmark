# Download and normalize NorGLM/NO-BoolQ for the yes/no tier.

"""
Input:  HuggingFace dataset NorGLM/NO-BoolQ (train / val / test .jsonl with
        keys {idx, passage, question, label}).

Output: processed/noboolq/{train,val,test}.jsonl with normalized schema:

    {
        "id": str,         # "noboolq-{split}-{idx}"
        "question": str,
        "context": str,
        "label": bool,     # True == "Ja", False == "Nei"
        "answer_text": "Ja" | "Nei",
        "tier": "yes_no",
        "source": "no_boolq",
        "split": "train" | "val" | "test",
    }
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PROC_NOBOOLQ

HF_BASE = "https://huggingface.co/datasets/NorGLM/NO-BoolQ/resolve/main"
SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[noboolq] cached {dst.name}")
        return
    print(f"[noboolq] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "norqa-pipeline/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dst.write_bytes(r.read())
    print(f"[noboolq] wrote {dst} ({dst.stat().st_size} bytes)")


def _normalize_rows(raw_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = bool(obj["label"])
            rows.append({
                "id": f"noboolq-{split}-{obj['idx']}",
                "question": obj["question"],
                "context": obj["passage"],
                "label": label,
                "answer_text": "Ja" if label else "Nei",
                "tier": "yes_no",
                "source": "no_boolq",
                "split": split,
            })
    return rows


def main() -> None:
    PROC_NOBOOLQ.mkdir(parents=True, exist_ok=True)
    raw_dir = PROC_NOBOOLQ / "_raw"
    summary: dict[str, dict] = {}
    for split, fname in SPLIT_FILES.items():
        url = f"{HF_BASE}/{fname}"
        raw = raw_dir / fname
        _download(url, raw)
        rows = _normalize_rows(raw, split)
        out = PROC_NOBOOLQ / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter(r["label"] for r in rows)
        summary[split] = {"n": len(rows), "yes": c[True], "no": c[False]}
        print(f"[noboolq] {split}: {len(rows)} (yes={c[True]}, no={c[False]}) -> {out}")
    (PROC_NOBOOLQ / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("[noboolq] done")


if __name__ == "__main__":
    main()

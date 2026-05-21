# Fine-tune a Norwegian/multilingual encoder

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import CKPTS, FIGS, LOGS, PREDS, SEED
from src.data.loader import load_tier

import transformers  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
transformers.logging.set_verbosity_warning()


def slug(name: str) -> str:
    return name.replace("/", "_")


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p = _norm(pred).split(); g = _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(p); rc = n / len(g)
    return 2 * pr * rc / (pr + rc)


def _safe_f1(tp, pp, gp):
    if pp == 0 or gp == 0:
        return 0.0
    p, r = tp / pp, tp / gp
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ===================== BoolQ branch =====================

def prepare_boolq_batches(rows, tok, max_len=384):
    feats = []
    for r in rows:
        enc = tok(
            r["question"], r["context"],
            truncation="only_second", max_length=max_len,
            return_attention_mask=True, return_token_type_ids=True,
        )
        feats.append({
            "id": r["id"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids") or [0] * len(enc["input_ids"]),
            "label": 1 if r["label"] else 0,
            "question": r["question"],
            "context": r["context"],
            "answer_text": r["answer_text"],
        })
    return feats


def pad_collate_boolq(batch, pad_id=0):
    L = max(len(b["input_ids"]) for b in batch)
    def pad(seqs, fill):
        return torch.tensor([s + [fill] * (L - len(s)) for s in seqs], dtype=torch.long)
    return {
        "input_ids": pad([b["input_ids"] for b in batch], pad_id),
        "attention_mask": pad([b["attention_mask"] for b in batch], 0),
        "token_type_ids": pad([b["token_type_ids"] for b in batch], 0),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "meta": batch,
    }


def train_boolq(args, train, val, test, device):
    tag = f"encoder__{slug(args.model)}__boolq"
    ck_dir = CKPTS / tag; ck_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        # Load the previously-trained checkpoint and tokenizer from disk.
        tok = AutoTokenizer.from_pretrained(ck_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ck_dir).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2,
        ).to(device)
    tr = prepare_boolq_batches(train, tok, args.max_len)
    va = prepare_boolq_batches(val, tok, args.max_len)
    te = prepare_boolq_batches(test, tok, args.max_len)
    print(f"[encoder/boolq] train={len(tr)} val={len(va)} test={len(te)} pad_id={tok.pad_token_id}")
    pad_id = tok.pad_token_id or 0

    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True,
                       collate_fn=lambda b: pad_collate_boolq(b, pad_id))
    dl_va = DataLoader(va, batch_size=args.bs * 2, shuffle=False,
                       collate_fn=lambda b: pad_collate_boolq(b, pad_id))
    dl_te = DataLoader(te, batch_size=args.bs * 2, shuffle=False,
                       collate_fn=lambda b: pad_collate_boolq(b, pad_id))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(dl_tr) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    log_path = LOGS / f"{tag}.json"
    log = {"args": vars(args), "step": [], "epoch": []}
    if args.eval_only and log_path.exists():
        # preserve the original training trace
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            pass
    best = -1.0

    @torch.no_grad()
    def eval_loop(dl):
        model.eval()
        correct = 0; n = 0
        pos_p = pos_g = pos_c = 0
        neg_p = neg_g = neg_c = 0
        preds = []
        for batch in dl:
            inp = {k: v.to(device) for k, v in batch.items() if k != "meta"}
            out = model(**{k: v for k, v in inp.items() if k != "labels"})
            p = out.logits.argmax(-1).cpu().tolist()
            g = batch["labels"].tolist()
            for i, (pi, gi) in enumerate(zip(p, g)):
                m = batch["meta"][i]
                correct += int(pi == gi); n += 1
                preds.append({"id": m["id"], "tier": "yes_no",
                          "question": m["question"],
                          "gold": "Ja" if gi == 1 else "Nei",
                          "pred": "Ja" if pi == 1 else "Nei"})
                if pi == 1: pos_p += 1
                if gi == 1: pos_g += 1
                if pi == gi == 1: pos_c += 1
                if pi == 0: neg_p += 1
                if gi == 0: neg_g += 1
                if pi == gi == 0: neg_c += 1
        macro = (_safe_f1(pos_c, pos_p, pos_g) + _safe_f1(neg_c, neg_p, neg_g)) / 2
        return correct / max(n, 1), macro, preds

    step = 0
    if not args.eval_only:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            for batch in dl_tr:
                inp = {k: v.to(device) for k, v in batch.items() if k != "meta"}
                out = model(**inp)
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                if step % 50 == 0:
                    log["step"].append({"epoch": epoch, "step": step, "loss": float(out.loss)})
                    print(f"[encoder/boolq] e{epoch} step{step} loss={out.loss.item():.4f}")
                step += 1
            acc, macro, _ = eval_loop(dl_va)
            log["epoch"].append({"epoch": epoch, "step_end": step,
                                 "val_acc": acc, "val_macroF1": macro})
            print(f"[encoder/boolq] e{epoch} val acc={acc:.4f} macroF1={macro:.4f} ({time.time()-t0:.1f}s)")
            if macro > best:
                best = macro
                model.save_pretrained(ck_dir); tok.save_pretrained(ck_dir)
                print(f"[encoder/boolq] new best -> {ck_dir}")
            log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

        model = AutoModelForSequenceClassification.from_pretrained(ck_dir).to(device)
    te_acc, te_macro, preds = eval_loop(dl_te)
    log["test"] = {"acc": te_acc, "macroF1": te_macro}
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    with open(PREDS / f"{tag}__test.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[encoder/boolq] TEST acc={te_acc:.4f} macroF1={te_macro:.4f}")
    save_bert_attention_examples(model, tok, va, device, FIGS / "attention" / tag, n=6,
                                  task="boolq")


# ===================== QA branch =====================

def prepare_qa_features(rows, tok, max_len=384, doc_stride=128, is_train=True):
    """Apply HF-standard SQuAD feature prep: windowed contexts + start/end labels."""
    feats = []
    for r in rows:
        q = r["question"].lstrip()
        c = r["context"]
        enc = tok(
            q, c,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        offsets_all = enc.pop("offset_mapping")
        sample_mapping = enc.pop("overflow_to_sample_mapping")
        for i in range(len(enc["input_ids"])):
            input_ids = enc["input_ids"][i]
            seq_ids = enc.sequence_ids(i)
            sample_idx = sample_mapping[i] if isinstance(sample_mapping, list) else int(sample_mapping[i])
            offsets = offsets_all[i]
            feat = {
                "id": r["id"],
                "input_ids": input_ids,
                "attention_mask": enc["attention_mask"][i],
                "token_type_ids": enc.get("token_type_ids", [[0] * len(input_ids)])[i]
                                  if enc.get("token_type_ids") else [0] * len(input_ids),
                "offset_mapping": [(o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets)],
                "context": c,
                "question": r["question"],
                "gold_answers": [a for a in r["answers"]["text"]],
                "gold_starts": [s for s in r["answers"]["answer_start"]],
                "feat_idx": i,
            }
            if is_train:
                if not r["answers"]["text"]:
                    feat["start_position"] = 0
                    feat["end_position"] = 0
                else:
                    a_start = r["answers"]["answer_start"][0]
                    a_end = a_start + len(r["answers"]["text"][0])
                    # find token range
                    tok_start = 0
                    while tok_start < len(seq_ids) and seq_ids[tok_start] != 1:
                        tok_start += 1
                    tok_end = len(seq_ids) - 1
                    while tok_end >= 0 and seq_ids[tok_end] != 1:
                        tok_end -= 1
                    if (tok_start >= len(offsets) or tok_end < 0 or
                        offsets[tok_start][0] > a_end or offsets[tok_end][1] < a_start):
                        feat["start_position"] = 0
                        feat["end_position"] = 0
                    else:
                        while tok_start < len(offsets) and offsets[tok_start][0] <= a_start:
                            tok_start += 1
                        feat["start_position"] = tok_start - 1
                        while tok_end >= 0 and offsets[tok_end][1] >= a_end:
                            tok_end -= 1
                        feat["end_position"] = tok_end + 1
            feats.append(feat)
    return feats


def pad_collate_qa(batch, pad_id, is_train):
    L = max(len(b["input_ids"]) for b in batch)
    def pad(key, fill):
        return torch.tensor([b[key] + [fill] * (L - len(b[key])) for b in batch], dtype=torch.long)
    out = {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "token_type_ids": pad("token_type_ids", 0),
        "meta": batch,
    }
    if is_train:
        out["start_positions"] = torch.tensor([b["start_position"] for b in batch], dtype=torch.long)
        out["end_positions"]   = torch.tensor([b["end_position"]   for b in batch], dtype=torch.long)
    return out


def decode_qa(features, all_start_logits, all_end_logits, n_best=20, max_ans=30):
    """SQuAD-style span decoding over windowed features. Returns id -> best pred dict.

    The returned dict carries question and gold so downstream prediction
    files can be analysed per question-type/source without re-joining with
    the dataset.
    """
    feats_per_id: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(features):
        feats_per_id[f["id"]].append(i)
    preds: dict[str, dict] = {}
    for qid, fids in feats_per_id.items():
        f0 = features[fids[0]]
        best = None
        for fi in fids:
            f = features[fi]
            s_logits = all_start_logits[fi]
            e_logits = all_end_logits[fi]
            top_s = np.argsort(s_logits)[-1: -n_best - 1: -1]
            top_e = np.argsort(e_logits)[-1: -n_best - 1: -1]
            offsets = f["offset_mapping"]
            for s in top_s:
                for e in top_e:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_ans:
                        continue
                    score = float(s_logits[s] + e_logits[e])
                    text = f["context"][offsets[s][0]: offsets[e][1]]
                    if not best or score > best["score"]:
                        best = {"score": score, "text": text}
        gold = f0["gold_answers"][0] if f0["gold_answers"] else ""
        out = {
            "score": (best or {}).get("score", 0.0),
            "text": (best or {}).get("text", ""),
            "gold": gold,
            "question": f0["question"],
        }
        preds[qid] = out
    return preds


def train_qa(args, train, val, test, device):
    tag = f"encoder__{slug(args.model)}__{args.task}"
    ck_dir = CKPTS / tag; ck_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(ck_dir)
        model = AutoModelForQuestionAnswering.from_pretrained(ck_dir).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    pad_id = tok.pad_token_id or 0
    feats_tr = prepare_qa_features(train, tok, args.max_len, args.doc_stride, is_train=True)
    feats_va = prepare_qa_features(val, tok, args.max_len, args.doc_stride, is_train=False)
    feats_te = prepare_qa_features(test, tok, args.max_len, args.doc_stride, is_train=False)
    print(f"[encoder/qa] features: train={len(feats_tr)} val={len(feats_va)} test={len(feats_te)}")
    dl_tr = DataLoader(feats_tr, batch_size=args.bs, shuffle=True,
                       collate_fn=lambda b: pad_collate_qa(b, pad_id, is_train=True))
    dl_va = DataLoader(feats_va, batch_size=args.bs * 2, shuffle=False,
                       collate_fn=lambda b: pad_collate_qa(b, pad_id, is_train=False))
    dl_te = DataLoader(feats_te, batch_size=args.bs * 2, shuffle=False,
                       collate_fn=lambda b: pad_collate_qa(b, pad_id, is_train=False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(dl_tr) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    log_path = LOGS / f"{tag}.json"
    log = {"args": vars(args), "step": [], "epoch": []}
    if args.eval_only and log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            pass
    best = -1.0

    tier_name = {"squad_short": "short_factual", "squad_long": "long_span",
                 "squad_all": "extractive_all"}.get(args.task, "extractive")

    @torch.no_grad()
    def eval_loop(feats, dl):
        model.eval()
        starts, ends = [], []
        for batch in dl:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            starts.append(out.start_logits.cpu().numpy())
            ends.append(out.end_logits.cpu().numpy())
        starts = np.concatenate(starts, 0)
        ends = np.concatenate(ends, 0)
        preds_by_id = decode_qa(feats, starts, ends)
        em = f1 = 0; n = 0
        preds_list = []
        for qid, p in preds_by_id.items():
            em += int(_norm(p["text"]) == _norm(p["gold"]))
            f1 += _f1(p["text"], p["gold"])
            n += 1
            preds_list.append({
                "id": qid,
                "tier": tier_name,
                "question": p["question"],
                "pred": p["text"],
                "gold": p["gold"],
            })
        return em / max(n, 1), f1 / max(n, 1), preds_list

    step = 0
    if not args.eval_only:
        for epoch in range(args.epochs):
            model.train(); t0 = time.time()
            for batch in dl_tr:
                inp = {k: v.to(device) for k, v in batch.items() if k != "meta"}
                out = model(**inp)
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                if step % 50 == 0:
                    log["step"].append({"epoch": epoch, "step": step, "loss": float(out.loss)})
                    print(f"[encoder/qa] e{epoch} step{step} loss={out.loss.item():.4f}")
                step += 1
            va_em, va_f1, _ = eval_loop(feats_va, dl_va)
            log["epoch"].append({"epoch": epoch, "step_end": step,
                                 "val_em": va_em, "val_f1": va_f1})
            print(f"[encoder/qa] e{epoch} val EM={va_em:.4f} F1={va_f1:.4f} ({time.time()-t0:.1f}s)")
            if va_f1 > best:
                best = va_f1
                model.save_pretrained(ck_dir); tok.save_pretrained(ck_dir)
            log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

        model = AutoModelForQuestionAnswering.from_pretrained(ck_dir).to(device)
    te_em, te_f1, preds = eval_loop(feats_te, dl_te)
    log["test"] = {"em": te_em, "f1": te_f1}
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    with open(PREDS / f"{tag}__test.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[encoder/qa] TEST EM={te_em:.4f} F1={te_f1:.4f}")
    save_bert_attention_examples(model, tok, val[:8], device, FIGS / "attention" / tag, n=6,
                                  task="qa")


# ===================== Attention viz =====================

@torch.no_grad()
def save_bert_attention_examples(model, tok, sample_rows, device, out_dir: Path,
                                  n: int = 6, task: str = "qa") -> None:
    """Average attention from the last layer across all heads, plot as heatmap.

    Plots how each question token attends to context tokens — the closest BERT
    analog of BiDAF's bidirectional attention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_rows = list(sample_rows)[:n]
    for idx, item in enumerate(sample_rows):
        if isinstance(item, dict) and "question" in item and "context" in item:
            q = item["question"]; c = item["context"]
            gold = item.get("answers", {}).get("text", [item.get("answer_text", "")])[0] \
                if isinstance(item.get("answers"), dict) else item.get("answer_text", "")
        else:
            continue
        enc = tok(q, c, return_tensors="pt", truncation=True, max_length=384,
                  return_token_type_ids=True).to(device)
        out = model.base_model(**enc, output_attentions=True, return_dict=True)
        # last layer, mean over heads
        att = out.attentions[-1].mean(dim=1)[0].cpu().numpy()   # (T, T)
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
        # split q/c by token_type_ids (XLM-R has no token_type; use [SEP] heuristic)
        tti = enc.get("token_type_ids", None)
        if tti is not None:
            tti = tti[0].cpu().tolist()
            q_mask = [t == 0 for t in tti]
            c_mask = [t == 1 for t in tti]
        else:
            sep = tok.sep_token or "</s>"
            seps = [i for i, t in enumerate(toks) if t == sep]
            if len(seps) < 2:
                continue
            q_mask = [False] * len(toks); c_mask = [False] * len(toks)
            for i in range(seps[0]):
                q_mask[i] = True
            for i in range(seps[0] + 1, seps[1]):
                c_mask[i] = True

        q_idx = [i for i, m in enumerate(q_mask) if m]
        c_idx = [i for i, m in enumerate(c_mask) if m]
        if not q_idx or not c_idx:
            continue
        att_qc = att[np.ix_(q_idx, c_idx)]
        # truncate for readability
        if len(c_idx) > 80:
            att_qc = att_qc[:, :80]; c_idx = c_idx[:80]
        # row-norm
        att_qc = att_qc / (att_qc.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(max(6, len(c_idx) * 0.18),
                                         max(3, len(q_idx) * 0.3)))
        im = ax.imshow(att_qc, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(q_idx)))
        ax.set_yticklabels([toks[i].replace("##", "") for i in q_idx], fontsize=8)
        ax.set_xticks(range(len(c_idx)))
        ax.set_xticklabels([toks[i].replace("##", "") for i in c_idx],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(f"BERT attn (last layer, mean over heads)\nQ: {q[:80]}\nA: {gold[:80]}",
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.025)
        plt.tight_layout()
        plt.savefig(out_dir / f"ex{idx:02d}.png", dpi=140)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["boolq", "squad_short", "squad_long", "squad_all"])
    p.add_argument("--model", default="NbAiLab/nb-bert-large")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load the existing checkpoint and only "
                        "re-run test-set eval. Used to refresh prediction files "
                        "after scoring changes.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)
    print(f"[encoder] device={device} model={args.model} task={args.task}")

    tier_map = {"boolq": "yes_no", "squad_short": "short_factual",
                "squad_long": "long_span", "squad_all": "extractive_all"}
    train, val, test = load_tier(tier_map[args.task])

    for d in (LOGS, PREDS, CKPTS):
        d.mkdir(parents=True, exist_ok=True)

    if args.task == "boolq":
        train_boolq(args, train, val, test, device)
    else:
        train_qa(args, train, val, test, device)


if __name__ == "__main__":
    main()

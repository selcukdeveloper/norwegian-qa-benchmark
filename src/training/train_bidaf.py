### Train BiDAF (Seo et al. 2017)

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import CKPTS, FIGS, LOGS, PREDS, SEED
from src.data.loader import load_tier
from src.models.bidaf import BiDAF, BiDAFConfig

PAD, UNK = "<pad>", "<unk>"
SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in SPLIT_RE.finditer(text)]


def build_vocab(rows: list[dict], min_freq: int = 2, max_size: int = 60_000) -> dict[str, int]:
    counter: Counter = Counter()
    for r in rows:
        for t, _ in tokenize(r["question"]):
            counter[t.lower()] += 1
        for t, _ in tokenize(r["context"]):
            counter[t.lower()] += 1
    vocab = {PAD: 0, UNK: 1}
    for tok, c in counter.most_common(max_size - 2):
        if c < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab


def char_offset_to_token(answer_start: int, answer_text: str, tokens: list[tuple[str, int]]):
    if answer_start < 0 or not answer_text:
        return None
    answer_end = answer_start + len(answer_text)
    s_tok = e_tok = None
    for i, (_, off) in enumerate(tokens):
        if s_tok is None and off >= answer_start:
            s_tok = max(0, i - 1) if off > answer_start else i
        if off >= answer_end:
            e_tok = i - 1
            break
    if s_tok is None:
        return None
    if e_tok is None:
        e_tok = len(tokens) - 1
    e_tok = max(e_tok, s_tok)
    return s_tok, e_tok


class ExtractiveDataset(Dataset):
    def __init__(self, rows, vocab, max_ctx=400, max_q=40):
        self.vocab = vocab
        self.max_ctx = max_ctx
        self.max_q = max_q
        self.items = []
        for r in rows:
            ctx_toks = tokenize(r["context"])
            q_toks = tokenize(r["question"])
            if len(ctx_toks) == 0 or len(q_toks) == 0:
                continue
            gold = r["answers"]["text"][0]
            astart = r["answers"]["answer_start"][0]
            span = char_offset_to_token(astart, gold, ctx_toks)
            if span is None:
                continue
            s, e = span
            if s >= max_ctx or e >= max_ctx:
                continue # skip examples where answer is truncated out of context
            self.items.append({
                "id": r["id"],
                "ctx_toks": ctx_toks[:max_ctx],
                "q_toks": q_toks[:max_q],
                "start": s,
                "end": e,
                "context": r["context"],
                "question": r["question"],
                "answer": gold,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        ctx_ids = [self.vocab.get(t.lower(), 1) for t, _ in it["ctx_toks"]]
        q_ids = [self.vocab.get(t.lower(), 1) for t, _ in it["q_toks"]]
        return {
            "id": it["id"],
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "start": torch.tensor(it["start"], dtype=torch.long),
            "end": torch.tensor(it["end"], dtype=torch.long),
            "meta": {
                "ctx_toks": [t for t, _ in it["ctx_toks"]],
                "q_toks": [t for t, _ in it["q_toks"]],
                "context": it["context"],
                "question": it["question"],
                "answer": it["answer"],
            },
        }


class BoolQDataset(Dataset):
    def __init__(self, rows, vocab, max_ctx=400, max_q=40):
        self.vocab = vocab
        self.max_ctx = max_ctx
        self.max_q = max_q
        self.items = []
        for r in rows:
            ctx = tokenize(r["context"])
            q = tokenize(r["question"])
            if not ctx or not q:
                continue
            self.items.append({
                "id": r["id"],
                "ctx_toks": ctx[:max_ctx],
                "q_toks": q[:max_q],
                "label": 1 if r["label"] else 0,
                "question": r["question"],
                "context": r["context"],
                "answer": r["answer_text"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        ctx_ids = [self.vocab.get(t.lower(), 1) for t, _ in it["ctx_toks"]]
        q_ids = [self.vocab.get(t.lower(), 1) for t, _ in it["q_toks"]]
        return {
            "id": it["id"],
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "label": torch.tensor(it["label"], dtype=torch.long),
            "meta": {
                "ctx_toks": [t for t, _ in it["ctx_toks"]],
                "q_toks": [t for t, _ in it["q_toks"]],
                "question": it["question"],
                "context": it["context"],
                "answer": it["answer"],
            },
        }


def pad_collate(batch, pad_id=0):
    keys_pad = ["ctx_ids", "q_ids"]
    out = {}
    for k in keys_pad:
        seqs = [b[k] for b in batch]
        lens = [s.size(0) for s in seqs]
        L = max(lens) if lens else 1
        arr = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            arr[i, : s.size(0)] = s
        out[k] = arr
        out[k + "_mask"] = (arr != pad_id).long()
    for k in ("start", "end", "label"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    out["ids"] = [b["id"] for b in batch]
    out["metas"] = [b["meta"] for b in batch]
    return out


def init_emb_nbbert(vocab: dict[str, int], emb_dim: int) -> torch.Tensor:
    if emb_dim != 768:
        raise ValueError(
            f"--emb nbbert requires --emb-dim 768 (NB-BERT-base hidden size); got {emb_dim}"
        )
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")
    mdl = AutoModel.from_pretrained("NbAiLab/nb-bert-base")
    W = mdl.embeddings.word_embeddings.weight.detach().cpu()  # (V_bert, 768)
    mat = torch.empty(len(vocab), emb_dim)
    nn.init.normal_(mat, mean=0.0, std=0.02)
    mat[0].zero_()  # PAD
    hits = 0
    for word, idx in vocab.items():
        if word in (PAD, UNK):
            continue
        sub = tok.tokenize(word)
        if not sub:
            continue
        ids = tok.convert_tokens_to_ids(sub)
        mat[idx] = W[ids].mean(0)
        hits += 1
    print(f"[bidaf] nbbert init: {hits}/{len(vocab)} vocab tokens mapped")
    return mat


@torch.no_grad()
def evaluate_extractive(model, loader, device, tier="extractive", max_ans=30):
    model.eval()
    em = f1 = 0
    n = 0
    preds = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        start, end, _ = model(
            batch["ctx_ids"], batch["ctx_ids_mask"],
            batch["q_ids"], batch["q_ids_mask"],
        )
        s_logp = F.log_softmax(start, dim=-1)
        e_logp = F.log_softmax(end, dim=-1)
        # joint argmax over (i,j) with i<=j and j-i<max_ans
        score = s_logp.unsqueeze(2) + e_logp.unsqueeze(1)  # (B,T,T)
        T = score.size(1)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=0) * \
               torch.tril(torch.ones(T, T, device=device), diagonal=max_ans)
        score = score.masked_fill(mask == 0, -1e9)
        flat = score.view(score.size(0), -1).argmax(dim=-1)
        i_pred = (flat // T).tolist()
        j_pred = (flat % T).tolist()
        for k, meta in enumerate(batch["metas"]):
            toks = meta["ctx_toks"]
            s, e = i_pred[k], j_pred[k]
            pred = " ".join(toks[s:e + 1])
            gold = meta["answer"]
            em += int(_norm(pred) == _norm(gold))
            f1 += _f1(pred, gold)
            n += 1
            preds.append({
                "id": batch["ids"][k],
                "tier": tier,
                "question": meta["question"],
                "gold": gold,
                "pred": pred,
                "start": s,
                "end": e,
            })
    return em / max(n, 1), f1 / max(n, 1), preds


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    nsame = sum(common.values())
    if nsame == 0:
        return 0.0
    prec = nsame / len(p)
    rec = nsame / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate_boolq(model, loader, device, tier="yes_no"):
    model.eval()
    correct = 0
    n = 0
    preds = []
    pos_pred = pos_gold = pos_correct = 0
    neg_pred = neg_gold = neg_correct = 0
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits, _ = model(
            batch_t["ctx_ids"], batch_t["ctx_ids_mask"],
            batch_t["q_ids"], batch_t["q_ids_mask"],
        )
        pred = logits.argmax(dim=-1)
        for k, meta in enumerate(batch["metas"]):
            p = int(pred[k].item())
            g = int(batch_t["label"][k].item())
            correct += int(p == g)
            n += 1
            preds.append({
                "id": batch["ids"][k],
                "tier": tier,
                "question": meta["question"],
                "gold": "Ja" if g == 1 else "Nei",
                "pred": "Ja" if p == 1 else "Nei",
            })
            if p == 1: pos_pred += 1
            if g == 1: pos_gold += 1
            if p == g == 1: pos_correct += 1
            if p == 0: neg_pred += 1
            if g == 0: neg_gold += 1
            if p == g == 0: neg_correct += 1
    acc = correct / max(n, 1)
    f1_pos = _safe_f1(pos_correct, pos_pred, pos_gold)
    f1_neg = _safe_f1(neg_correct, neg_pred, neg_gold)
    macro = (f1_pos + f1_neg) / 2
    return acc, macro, preds


def _safe_f1(tp, pp, gp) -> float:
    if pp == 0 or gp == 0:
        return 0.0
    p = tp / pp
    r = tp / gp
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def save_attention_examples(model, dataset, device, out_dir: Path, n: int = 6) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    indices = list(range(min(n, len(dataset))))
    for idx in indices:
        item = dataset[idx]
        b = pad_collate([item])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            out = model(b["ctx_ids"], b["ctx_ids_mask"], b["q_ids"], b["q_ids_mask"])
            S = out[-1].cpu().numpy()[0]  # (T, J)
        meta = b["metas"][0]
        T = sum(b["ctx_ids_mask"][0].cpu().tolist())
        J = sum(b["q_ids_mask"][0].cpu().tolist())
        S = S[:T, :J]
        ctx_t = meta["ctx_toks"][:T]
        q_t = meta["q_toks"][:J]
        S_show = np.exp(S - S.max(axis=1, keepdims=True))
        S_show = S_show / S_show.sum(axis=1, keepdims=True)
        if T > 80:
            ctx_t = ctx_t[:80]
            S_show = S_show[:80]
        fig, ax = plt.subplots(figsize=(max(6, J * 0.4), max(4, len(ctx_t) * 0.18)))
        im = ax.imshow(S_show, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(q_t)))
        ax.set_xticklabels(q_t, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ctx_t)))
        ax.set_yticklabels(ctx_t, fontsize=7)
        ax.set_title(f"BiDAF C2Q attention\nQ: {meta['question']}\nA: {meta.get('answer', '?')}",
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.025)
        plt.tight_layout()
        plt.savefig(out_dir / f"ex{idx:02d}.png", dpi=140)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["boolq", "squad_short", "squad_long", "squad_all"])
    p.add_argument("--emb", choices=["nbbert", "scratch"], default="scratch",
                   help="scratch = random init trained end-to-end (default); "
                        "nbbert = init from NB-BERT-base subword embeddings "
                        "(requires --emb-dim 768).")
    p.add_argument("--emb-dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max-ctx", type=int, default=400)
    p.add_argument("--max-q", type=int, default=40)
    p.add_argument("--freeze-emb", action="store_true", default=False,
                   help="If set, freezes the embedding table. Default is to "
                        "train embeddings end-to-end -- freezing on top of "
                        "scratch or nbbert init kept BiDAF from learning.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    set_seed(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)
    print(f"[bidaf] device={device}")

    tier_map = {
        "boolq": "yes_no",
        "squad_short": "short_factual",
        "squad_long": "long_span",
        "squad_all": "extractive_all",
    }
    train, val, test = load_tier(tier_map[args.task])
    print(f"[bidaf] task={args.task} train={len(train)} val={len(val)} test={len(test)}")

    vocab = build_vocab(train)
    print(f"[bidaf] |V|={len(vocab)}")

    is_boolq = args.task == "boolq"
    DSCls = BoolQDataset if is_boolq else ExtractiveDataset
    ds_tr = DSCls(train, vocab, args.max_ctx, args.max_q)
    ds_va = DSCls(val, vocab, args.max_ctx, args.max_q)
    ds_te = DSCls(test, vocab, args.max_ctx, args.max_q)
    print(f"[bidaf] usable train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, collate_fn=pad_collate)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, collate_fn=pad_collate)

    cfg = BiDAFConfig(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        task="boolq" if is_boolq else "extractive",
    )
    model = BiDAF(cfg).to(device)

    if args.emb == "nbbert":
        vecs = init_emb_nbbert(vocab, args.emb_dim)
        model.init_embeddings(vecs.to(device), freeze=args.freeze_emb)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tag = f"bidaf__{args.task}"
    LOGS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    CKPTS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}.json"
    ck_dir = CKPTS / tag
    ck_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "step": [], "epoch": []}

    best = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in dl_tr:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if is_boolq:
                logits, _ = model(
                    batch["ctx_ids"], batch["ctx_ids_mask"],
                    batch["q_ids"], batch["q_ids_mask"],
                )
                loss = F.cross_entropy(logits, batch["label"])
            else:
                s, e, _ = model(
                    batch["ctx_ids"], batch["ctx_ids_mask"],
                    batch["q_ids"], batch["q_ids_mask"],
                )
                loss = (F.cross_entropy(s, batch["start"]) + F.cross_entropy(e, batch["end"])) / 2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step % 50 == 0:
                print(f"[bidaf] e{epoch} step{step} loss={loss.item():.4f}")
                log["step"].append({"epoch": epoch, "step": step, "loss": float(loss)})
            step += 1
        sched.step()

        tier_name = tier_map[args.task]
        if is_boolq:
            va_acc, va_macro, _ = evaluate_boolq(model, dl_va, device, tier=tier_name)
            metric = va_macro
            print(f"[bidaf] e{epoch} val acc={va_acc:.4f} macroF1={va_macro:.4f}  ({time.time()-t0:.1f}s)")
            log["epoch"].append({"epoch": epoch, "step_end": step,
                                 "val_acc": va_acc, "val_macroF1": va_macro})
        else:
            va_em, va_f1, _ = evaluate_extractive(model, dl_va, device, tier=tier_name)
            metric = va_f1
            print(f"[bidaf] e{epoch} val EM={va_em:.4f} F1={va_f1:.4f}  ({time.time()-t0:.1f}s)")
            log["epoch"].append({"epoch": epoch, "step_end": step,
                                 "val_em": va_em, "val_f1": va_f1})

        if metric > best:
            best = metric
            torch.save({"model": model.state_dict(), "vocab": vocab, "cfg": vars(cfg)},
                       ck_dir / "best.pt")
            print(f"[bidaf] new best -> {ck_dir/'best.pt'}")
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    # Final test
    state = torch.load(ck_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    tier_name = tier_map[args.task]
    if is_boolq:
        te_acc, te_macro, preds = evaluate_boolq(model, dl_te, device, tier=tier_name)
        log["test"] = {"acc": te_acc, "macroF1": te_macro}
        print(f"[bidaf] TEST acc={te_acc:.4f} macroF1={te_macro:.4f}")
    else:
        te_em, te_f1, preds = evaluate_extractive(model, dl_te, device, tier=tier_name)
        log["test"] = {"em": te_em, "f1": te_f1}
        print(f"[bidaf] TEST EM={te_em:.4f} F1={te_f1:.4f}")
    with open(PREDS / f"{tag}__test.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    # Attention figures from val set
    save_attention_examples(model, ds_va, device, FIGS / "attention" / tag, n=6)
    print(f"[bidaf] done. tag={tag}")


if __name__ == "__main__":
    main()

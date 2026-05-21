from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PREDS, SEED
from src.data.loader import load_tier

# Norwegian instructions per tier. Keep these short — the model is zero-shot,
# not instruction-tuned to a specific template.
INSTR = {
    "yes_no":        "Les teksten og svar med ett ord: «Ja» eller «Nei».",
    "short_factual": "Les teksten og gi et kort svar, hentet ordrett fra teksten.",
    "long_span":     "Les teksten og gi et utfyllende svar (1–3 setninger) basert på teksten.",
}

# Greedy generation budget per tier (in new tokens). long_span needs to be
# generous because the "thinking" variant often produces a short reasoning
# trace before the answer; if we cut too early we capture the trace, not the
# answer.
MAX_NEW = {"yes_no": 8, "short_factual": 24, "long_span": 192}

# How long demo contexts are allowed to be when we prepend them as few-shot
# examples. Keeps the prompt from blowing past the model's context window.
DEMO_MAX_CHARS = 700


def render_prompt(tier: str, question: str, context: str, demos: list[dict] | None = None) -> str:
    """Build the full prompt. If demos is given, prepend each as a Tekst/Spørsmål/Svar block."""
    head = f"Instruksjon: {INSTR[tier]}\n\n"
    body = ""
    if demos:
        for d in demos:
            ctx = d["context"]
            if len(ctx) > DEMO_MAX_CHARS:
                ctx = ctx[:DEMO_MAX_CHARS].rsplit(" ", 1)[0] + " …"
            body += (
                f"Tekst: {ctx}\n"
                f"Spørsmål: {d['question']}\n"
                f"Svar: {d['gold']}\n\n"
            )
    body += (
        f"Tekst: {context}\n"
        f"Spørsmål: {question}\n"
        f"Svar:"
    )
    return head + body

_ECHO_PATTERNS = [
    "instruksjon",
    "spørsmål:",
    "tekst:",
    "jeg skal",
    "jeg bør",
    "jeg må",
    "la meg",
    "først skal",
    "for å svare",
    "basert på teksten",
    "setninger basert",
    "1-3 setninger",
    "1–3 setninger",
    "1 – 3 setninger",
    "ett ord:",
    "«ja» eller «nei»",
    "ja eller nei",
]

# Strip these prefixes from a candidate answer line.
_PREFIX_RE = re.compile(
    r"^[\"«]?\s*(svar(?:et)?|konklusjon|sammenfattet)\s*[:\-–]\s*",
    re.IGNORECASE,
)

_BULLET = re.compile(r"^\s*(?:[1-9]\.\s+[A-ZÆØÅ]|[-*•]\s+[A-ZÆØÅ])")
_HEADER = re.compile(r"^(jeg|først|svar(?:et)?\b|teksten\b|instruksjon|tip(?:s)?\b|step\b)",
                     re.IGNORECASE)


def extract_answer(decoded: str) -> str:
    lines = [l.strip() for l in decoded.splitlines()]
    n = len(lines)
    for i, s in enumerate(lines):
        if not s:
            continue
        low = s.lower()
        # standalone marker on its own line -> grab the next content line
        if low.rstrip(":") in {"svar", "konklusjon", "sammenfattet", "svaret er"}:
            for j in range(i + 1, n):
                if lines[j]:
                    return _PREFIX_RE.sub("", lines[j]).strip().strip('"«»') or ""
            continue
        if _BULLET.match(s):
            continue
        if _HEADER.match(s):
            continue
        if any(pat in low for pat in _ECHO_PATTERNS):
            continue
        # short label-lines that end in ":" are introductions, not answers
        if s.endswith(":") and len(s.split()) <= 4:
            continue
        cleaned = _PREFIX_RE.sub("", s).strip().strip('"«»')
        if cleaned:
            return cleaned
    return ""


def gold_of(row: dict, tier: str) -> str:
    if tier == "yes_no":
        return row["answer_text"]
    return row["answers"]["text"][0]


def _model_slug(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.replace("-", "_")


def cfg_tag(decode_mode: str, few_shot: int) -> str:
    if few_shot > 0 and decode_mode == "logit_bias":
        return f"logit_bias_fewshot{few_shot}"
    if few_shot > 0:
        return f"fewshot{few_shot}"
    return decode_mode  # "free" or "logit_bias"


def sample_demos(train: list[dict], tier: str, k: int, seed: int = SEED) -> list[dict]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    if tier == "yes_no":
        ja = [r for r in train if r.get("answer_text", "").lower().startswith("ja")]
        nei = [r for r in train if r.get("answer_text", "").lower().startswith("nei")]
        rng.shuffle(ja); rng.shuffle(nei)
        out = []
        for i in range(k):
            pool = ja if i % 2 == 0 else nei
            if not pool:
                pool = ja or nei
            out.append(pool.pop(0))
    else:
        pool = list(train)
        rng.shuffle(pool)
        out = pool[:k]
    return [
        {"question": r["question"], "context": r["context"], "gold": gold_of(r, tier)}
        for r in out
    ]


# logit bias: force first-token answer to be Ja / Nei

_JA_FORMS  = ["Ja", " Ja", "ja", " ja", "JA", " JA"]
_NEI_FORMS = ["Nei", " Nei", "nei", " nei", "NEI", " NEI"]


def build_yesno_token_map(tokenizer) -> tuple[list[int], dict[int, str]]:
    id_to_class: dict[int, str] = {}
    for forms, klass in [(_JA_FORMS, "Ja"), (_NEI_FORMS, "Nei")]:
        for v in forms:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                id_to_class[ids[0]] = klass
    if not any(v == "Ja" for v in id_to_class.values()) or \
       not any(v == "Nei" for v in id_to_class.values()):
        raise RuntimeError(
            "Could not find single-token forms for both Ja and Nei in this tokenizer."
        )
    return sorted(id_to_class.keys()), id_to_class


class AllowOnlyTokensProcessor:
    def __init__(self, allowed_ids: list[int]):
        self.allowed = torch.tensor(allowed_ids, dtype=torch.long)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[..., self.allowed.to(scores.device)] = 0.0
        return scores + mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HuggingFace causal LM, e.g. norallm/normistral-11b-thinking")
    p.add_argument("--tier", required=True,
                   choices=["yes_no", "short_factual", "long_span"])
    p.add_argument("--decode-mode", choices=["free", "logit_bias"], default="free",
                   help="logit_bias is only valid for --tier yes_no.")
    p.add_argument("--few-shot", type=int, default=0,
                   help="Number of demonstrations to prepend from the train split (0 = zero-shot).")
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = p.parse_args()

    if args.decode_mode == "logit_bias" and args.tier != "yes_no":
        raise SystemExit("--decode-mode logit_bias only applies to --tier yes_no")

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ) if args.device == "auto" else torch.device(args.device)
    cfg = cfg_tag(args.decode_mode, args.few_shot)
    print(f"[gen] device={device} model={args.model} tier={args.tier} cfg={cfg}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = True
    model.eval()

    train, _, test = load_tier(args.tier)
    demos = sample_demos(train, args.tier, args.few_shot) if args.few_shot > 0 else []
    print(f"[gen] test={len(test)} examples, demos={len(demos)}")

    # Logit-bias setup (only for yes_no).
    logits_processor = None
    id_to_class: dict[int, str] = {}
    if args.decode_mode == "logit_bias":
        allowed_ids, id_to_class = build_yesno_token_map(tok)
        logits_processor = [AllowOnlyTokensProcessor(allowed_ids)]
        print(f"[gen] logit_bias: {len(allowed_ids)} allowed first-token IDs")

    # Few-shot prompts are longer, so allow a roomier truncation budget.
    max_input_len = 3072 if args.few_shot > 0 else 1800
    # In logit_bias mode we generate exactly one new token (the answer);
    # otherwise use the tier's normal budget.
    max_new = 1 if args.decode_mode == "logit_bias" else MAX_NEW[args.tier]

    out_path = PREDS / f"zs__{_model_slug(args.model)}__{args.tier}__{cfg}__test.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    t0 = time.time()
    for i in tqdm(range(0, len(test), args.bs)):
        batch = test[i:i + args.bs]
        prompts = [render_prompt(args.tier, r["question"], r["context"], demos) for r in batch]
        enc = tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_len,
        ).to(device)
        # Decoder-only LMs (Mistral, GPT, etc.) don't accept token_type_ids;
        # only feed input_ids + attention_mask to generate().
        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            logits_processor=logits_processor,
        )
        for j, ids in enumerate(gen):
            in_len = enc["input_ids"][j].shape[0]
            new_ids = ids[in_len:]
            decoded = tok.decode(new_ids, skip_special_tokens=True).strip()
            if args.decode_mode == "logit_bias":
                # The first new token id is the constrained answer; map it to
                # the canonical class word so the scorer sees a clean Ja / Nei.
                first_id = int(new_ids[0].item()) if len(new_ids) > 0 else -1
                answer = id_to_class.get(first_id, decoded)
            else:
                answer = extract_answer(decoded)
            row = batch[j]
            fout.write(json.dumps({
                "id": row["id"],
                "tier": args.tier,
                "cfg": cfg,
                "question": row["question"],
                "gold": gold_of(row, args.tier),
                "pred": answer,
                "raw": decoded,
            }, ensure_ascii=False) + "\n")
    fout.close()
    print(f"[gen] wrote {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

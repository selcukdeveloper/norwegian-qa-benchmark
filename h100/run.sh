# Single-pass run that produces every model prediction we report:
#   1) BiDAF (attention baseline, no transformer) on the three tasks
#   2) mBERT and NB-BERT-base fine-tunes on the three tasks
#   3) NorMistral-11B-thinking zero-shot on the three tiers

# Lexical baselines and scoring are produced separately by `bash score.sh` after sync_down.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/logs/h100

run() {
    local name="$1"; shift
    echo "==[$(date -u +%H:%M:%S)] $name=="
    "$@" 2>&1 | tee "results/logs/h100/${name}.txt"
}

# BiDAF (attention only)
run bidaf_boolq        python3 -u -m src.training.train_bidaf --task boolq        --epochs 15 --bs 32 --lr 1e-3
run bidaf_squad_short  python3 -u -m src.training.train_bidaf --task squad_short  --epochs 15 --bs 32 --lr 1e-3
run bidaf_squad_long   python3 -u -m src.training.train_bidaf --task squad_long   --epochs 15 --bs 32 --lr 1e-3

# mBERT fine-tunes
run mbert_boolq        python3 -u -m src.training.train_encoder --task boolq        --model bert-base-multilingual-cased --bs 16 --epochs 3 --max-len 384
run mbert_squad_short  python3 -u -m src.training.train_encoder --task squad_short  --model bert-base-multilingual-cased --bs 16 --epochs 4 --max-len 384
run mbert_squad_long   python3 -u -m src.training.train_encoder --task squad_long   --model bert-base-multilingual-cased --bs 16 --epochs 4 --max-len 384

# NB-BERT-base fine-tunes
run nbbert_boolq       python3 -u -m src.training.train_encoder --task boolq        --model NbAiLab/nb-bert-base --bs 16 --epochs 3 --max-len 384
run nbbert_squad_short python3 -u -m src.training.train_encoder --task squad_short  --model NbAiLab/nb-bert-base --bs 16 --epochs 4 --max-len 384
run nbbert_squad_long  python3 -u -m src.training.train_encoder --task squad_long   --model NbAiLab/nb-bert-base --bs 16 --epochs 6 --max-len 384

# NorMistral-11B-thinking
MODEL=norallm/normistral-11b-thinking

for tier in yes_no short_factual long_span; do
    run "gen__free__${tier}" python3 -u -m src.eval.generate_predictions \
        --model "$MODEL" --tier "$tier" --bs 2
done

run "gen__logit_bias__yes_no" python3 -u -m src.eval.generate_predictions \
    --model "$MODEL" --tier yes_no --decode-mode logit_bias --bs 2

for tier in yes_no short_factual long_span; do
    run "gen__fewshot3__${tier}" python3 -u -m src.eval.generate_predictions \
        --model "$MODEL" --tier "$tier" --few-shot 3 --bs 2
done

echo "[h100] run complete."

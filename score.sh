# Local post-processing step.

# Produces the lexical baselines and then scores every prediction file under results/predictions/. Idempotent: safe to re-run after pulling new results from the remote workspace.

set -euo pipefail
cd "$(dirname "$0")"

# Lexical baselines
python3 -m src.eval.baselines --which all

# Score every prediction file. BERTScore is requested only for long_span, set BERTSCORE=0 to skip it entirely.
BERTSCORE="${BERTSCORE:-1}"
for f in results/predictions/*.jsonl; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .jsonl)
    extra=""
    if [[ "$base" == *long_span* || "$base" == *squad_long* ]] && [[ "$BERTSCORE" == "1" ]]; then
        extra="--bertscore"
    fi
    echo "[score] $base"
    python3 -m src.eval.score --pred "$f" $extra > /dev/null
done

echo "[score] done. metrics -> results/metrics/"

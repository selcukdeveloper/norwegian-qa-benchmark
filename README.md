# Norwegian Question & Answering Benchmark

This repository contains data pipelines for experimenting with
question-answering on Norwegian text. The goal is to compare a few
models on three answer types (yes/no, short factual spans, and longer spans)
and make it easy to reproduce training, scoring, and figures.

Quick Guide
- Prepare processed data (one-time): run the scripts in `src/data/`.
- Train models on the workspace (remote): `h100/run.sh` handles training
  and generation.
- Pull results back to your local: `h100/sync_down.sh`, then run `bash score.sh`
  to compute metrics and baselines.
- Produce figures locally with `python3 -m src.viz.dataset_figures` and
  `python3 -m src.viz.results_figures`.

Main folders
- NorQuAD data/: original raw NorQuAD JSON
- processed/: tiered jsonl files used by training/eval (regenerate via `src/data`)
- src/: processing, models, training loops, evaluation, and plotting code
- h100/: helper scripts for running on the remote workspace
- results/: checkpoints, predictions, metrics, figures

How to Run?

```bash
# Build processed data (run once or when raw data changes):
python3 -m src.data.process_norquad
python3 -m src.data.process_noboolq

# On your workspace (run once)
source h100/setup.sh 

# On your workspace (run every session)
conda activate norqa
bash h100/run.sh

# Back on your local: pull results and score
bash h100/sync_down.sh
bash score.sh

# Make figures locally:
python3 -m src.viz.dataset_figures
python3 -m src.viz.results_figures
```

Notes and tips
- `score.sh` is safe to re-run when new prediction files arrive.
- BERTScore can be longer on first run (downloads XLM-R weights). Set
  `BERTSCORE=0` to skip it.
- The yes/no scorer is strict about abstentions; see `src/eval/score.py` for
  the exact behavior.
# norwegian-qa-benchmark

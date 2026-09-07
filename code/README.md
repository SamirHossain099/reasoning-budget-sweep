# Code

Reproduction code for the dataset in `../data/`.

| File | Purpose |
|---|---|
| `generate.py` | Generation harness. Resumable, memory-guarded, caches one row per response. |
| `data.py` | Benchmark loaders (AIME 2024/2025, MATH-500) |
| `verify.py` | Answer extraction and staged correctness checking |
| `test_verify.py` | 20 unit tests for the answer checker |
| `rescore.py` | Recompute correctness from cached answers without regenerating |
| `analyze_gen.py` | Aggregation: accuracy, answer rate, completion rate per cell |
| `reliability_gate.py` | Split-half reliability with length correction |
| `corrective_analysis.py` | Identity check, variance decomposition, split-half noise test |
| `status.py` | Sweep progress reporting |

Requires: torch (CUDA build), transformers, datasets, pandas, scipy, math_verify.

Example:

    python -m generate --models r1-1.5b --benchmarks math500 --budgets 4096 --seeds 0 1 2 --limit 100

Note `generate.py` forces `use_cache=True` after loading. Some fine-tuned checkpoints
(Open-RS1, Open-RS2, Open-RS3) ship `use_cache: false`, which disables the KV cache and makes
generation roughly 200x slower while inflating reserved memory until it spills to system RAM.

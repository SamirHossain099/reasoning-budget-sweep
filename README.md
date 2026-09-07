# Reasoning Budget Sweep: truncation and answer-production measurements for 12 small reasoning LLMs

A per-response measurement dataset covering 12 open reasoning language models (0.6B to 7B, mostly
sub-2B) evaluated on AIME 2024, AIME 2025 and MATH-500 across six generation token budgets
(512 to 16384) and three seeds. 11,118 scored responses, 38.7 million generated tokens.

The dataset records, for every single response, whether the model ran out of its token budget,
whether an answer could be extracted at all, and whether that answer was correct. It is intended
for meta-analysis of how generation budgets interact with benchmark scores in the small-model
regime, which most published evaluation-reliability work does not cover (comparable studies
typically start at 1.5B or 8B, or do not report model sizes).

## What this is, and what it is not

This is a **measurements** dataset, not a **generations** dataset.

**Included:** per-response token count, truncation flag, answer-extraction flag, the extracted
answer string, the gold answer, and the correctness label.

**Not included:** the full chain-of-thought text. Generations were scored on the fly and the text
was discarded to keep storage small. This is the dataset's main limitation and it rules out some
reuse, notably re-scoring with a different answer extractor, chain-of-thought faithfulness analysis,
and any study needing the reasoning trace itself. If you need the text, this dataset tells you the
exact configuration to regenerate it, but it does not contain it.

## Files

| File | Description |
|---|---|
| `data/reasoning_budget_sweep.parquet` | The dataset (recommended) |
| `data/reasoning_budget_sweep.csv` | Same data as CSV |
| `code/` | Generation harness, answer checker, and analysis used to produce it |

## Schema

One row per (model, benchmark, budget, seed, item).

| Column | Type | Meaning |
|---|---|---|
| `model` | str | Short model key used in this study |
| `model_hf_id` | str | Full Hugging Face repository id |
| `benchmark` | str | `aime24`, `aime25`, or `math500` |
| `item_id` | str | Problem identifier within the benchmark |
| `seed` | int | Sampling seed (0, 1, 2) |
| `budget` | int | `max_new_tokens` allowed for this response |
| `n_new_tokens` | int | Tokens actually generated |
| `reached_cap` | bool | True if generation hit the budget (was truncated) |
| `has_answer` | bool | True if an answer could be extracted |
| `answer` | str/null | The extracted answer, or null when none was found |
| `gold` | str | Reference answer |
| `correct` | bool | Whether `answer` matches `gold` |

Note the definitional relationship: `correct` implies `has_answer` in every row, so
`accuracy == answer_rate * P(correct | answer)` holds exactly by construction. Any analysis
regressing accuracy on answer rate is therefore partly tautological. This is stated explicitly
because it is easy to miss.

## How it was produced

- Sampling: temperature 0.6, top_p 0.95, three seeds per configuration.
- Prompting: each model's own chat template, with an instruction to place the final answer in
  `\boxed{}`.
- Truncation: `reached_cap` is true when the generated length equals the budget.
- Answer extraction: last `\boxed{...}` in the output, with a fallback pattern for integer answers.
- Correctness: staged check of normalised string equality, then numeric equality, then symbolic
  equality via `math_verify`. The checker was manually audited on a stratified sample (0 false
  positives in 12; 1 false negative in 20, a base-subscript formatting case, subsequently fixed)
  and is covered by 20 unit tests in `code/test_verify.py`.
- Hardware: a single NVIDIA RTX 5080 (16 GB), which is why the model range stops below 8B.

## Models

`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`,
`agentica-org/DeepScaleR-1.5B-Preview`, `knoveleng/Open-RS1`, `knoveleng/Open-RS2`,
`knoveleng/Open-RS3`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen2.5-Math-1.5B-Instruct`,
`Intelligent-Internet/II-Thought-1.5B-Preview`, `RUC-AIBOX/STILL-3-1.5B-preview`,
`nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`.

Coverage is not a complete grid. Budget 4096 is complete for all 11 sub-2B models; higher and lower
budgets cover fewer models. `r1-7b` was run only at budget 1024. Check counts before comparing
across cells, and note that two cells were excluded from some analyses for incomplete coverage.

## Benchmarks and provenance

- AIME 2024 from `Maxwell-Jia/AIME_2024`; AIME 2025 from `yentinglin/aime_2025`.
- MATH-500 from `HuggingFaceH4/MATH-500`, subsampled to a fixed 100-item subset (deterministic
  shuffle, seed 12345) to fit the compute budget. The full 500 items were not used.

Users should observe the licences of the upstream benchmarks and models. This release contains only
short extracted answer strings, not model output text.

## Licence

Code in `code/` is MIT. The data files are released under CC-BY-4.0.

## Citation

See `CITATION.cff`. Please cite the concept DOI, which always resolves to the latest version.

## Known limitations

1. No chain-of-thought text (see above).
2. Unbalanced coverage across budgets and models.
3. MATH-500 is a 100-item subsample, so per-model accuracy has wider intervals than the full set.
4. AIME has only 30 items per year; per-model accuracies carry a binomial standard error of roughly
   4 percentage points at three seeds. Rankings built from single cells are not reliable, and any
   ranking analysis on this data needs an explicit uncertainty treatment.
5. A single hardware and software stack, so no cross-implementation variance is captured.

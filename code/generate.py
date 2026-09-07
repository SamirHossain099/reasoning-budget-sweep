"""Generation harness for budget-swept reasoning evaluation. Pausable and resumable.

Sweep (model x benchmark x budget x seed), detect truncation, extract + check answers, cache
per-response rows. Designed to run in batches whenever the machine is idle:

  * Press Ctrl+C once -> the CURRENT batch finishes and is saved, then it exits cleanly.
  * Re-run the SAME command -> it resumes, skipping every item already generated (batch-granular),
    not just whole cells. You lose at most one in-flight batch on a hard kill.

Layout: results/generations/<model>/<bench>/b<budget>_s<seed>/part_XXXX.parquet (+ _DONE marker).
Legacy flat files b<budget>_s<seed>.parquet (from the first run) are recognised as completed.

Usage (resume-friendly; safe to re-run):
  python -m src.generate --models r1-1.5b --benchmarks aime24 aime25 \
      --budgets 1024 2048 4096 8192 16384 32768 --seeds 0 1 2
"""
from __future__ import annotations
import argparse, ctypes, gc, os, signal, sys, time
from pathlib import Path

# MUST be set before torch initialises CUDA. Without this the caching allocator reserves far more
# than it uses (measured: 5 GB used but 28-32 GB reserved at batch 30), which overflows the 16 GB
# card into Windows shared system RAM and collapses throughput to 1-4 tok/s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import data as datamod
from src.verify import extract_answer, check_answer

# Keep Windows from sleeping while the sweep runs (the run is suspended if the PC sleeps).
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED keeps the SYSTEM awake (display may still turn off).
_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def preflight_gpu(max_used_mib: int = 1500):
    """Refuse to start if another process is already holding VRAM: with less free memory the run
    silently spills to system RAM and crawls at 1-4 tok/s instead of erroring."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        used, total = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
    except Exception:
        return
    print(f"[preflight] GPU {used}/{total} MiB in use before load", flush=True)
    if used > max_used_mib:
        print(f"[preflight] WARNING: {used} MiB already in use by another process. "
              f"Close it (or kill stale python) or this run will be very slow.", flush=True)


def keep_awake(on: bool = True):
    try:
        flags = (_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) if on else _ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass  # non-Windows or restricted; harmless

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "generations"

MODELS = {
    "r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepscaler-1.5b": "agentica-org/DeepScaleR-1.5B-Preview",
    "openrs1-1.5b": "knoveleng/Open-RS1",
    "openrs3-1.5b": "knoveleng/Open-RS3",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    # added 2026-07-29 to raise n for the rank-instability Kendall tau (n=5 was underpowered).
    # All verified ungated; Open-RS2 ships use_cache=False (load() auto-corrects it).
    "openrs2-1.5b": "knoveleng/Open-RS2",
    "ii-thought-1.5b": "Intelligent-Internet/II-Thought-1.5B-Preview",
    "qwen25math-1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "still3-1.5b": "RUC-AIBOX/STILL-3-1.5B-preview",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "nemotron-1.5b": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
}

STOP = False  # set by SIGINT handler; loop stops after the current batch is saved


def _on_sigint(signum, frame):
    global STOP
    if not STOP:
        STOP = True
        print("\n[pause requested] finishing current batch, saving, then exiting cleanly... "
              "(re-run the same command to resume)", flush=True)


def load(model_key, device="cuda"):
    hf = MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(hf)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    try:
        m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float16).to(device)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float16).to(device)
    m.eval()
    # Some fine-tuned checkpoints (Open-RS1/RS3) ship `use_cache: false` left over from training.
    # Without the KV cache every decode step recomputes the entire sequence: ~200x slower, and the
    # churn inflates reserved memory (measured 28 GB vs 6 GB) until it spills to system RAM.
    if not getattr(m.config, "use_cache", True):
        print(f"    [fix] {model_key}: config had use_cache=False -> forcing True", flush=True)
    m.config.use_cache = True
    if getattr(m, "generation_config", None) is not None:
        m.generation_config.use_cache = True
    return m, tok


def cell_paths(mkey, bench, budget, seed):
    base = OUT / mkey / bench
    return base / f"b{budget}_s{seed}.parquet", base / f"b{budget}_s{seed}"  # legacy flat, cell dir


def done_item_ids(legacy, cell_dir) -> set:
    ids = set()
    if legacy.exists():
        ids |= set(pd.read_parquet(legacy)["item_id"].astype(str))
    if cell_dir.exists():
        for p in cell_dir.glob("part_*.parquet"):
            ids |= set(pd.read_parquet(p)["item_id"].astype(str))
    return ids


@torch.no_grad()
def gen_batch(model, tok, items, budget, seed, device):
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["question"]}],
                                       add_generation_prompt=True, tokenize=False) for it in items]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    torch.manual_seed(seed * 100003 + budget)
    out = model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=0.6,
                         top_p=0.95, pad_token_id=tok.pad_token_id)
    newtok = out[:, enc.input_ids.shape[1]:]
    rows = []
    for k, it in enumerate(items):
        seq = newtok[k]
        n_new = int((seq != tok.pad_token_id).sum().item())
        text = tok.decode(seq, skip_special_tokens=True)
        ans = extract_answer(text)
        rows.append({"model": None, "benchmark": it["benchmark"], "item_id": it["item_id"],
                     "seed": seed, "budget": budget, "n_new_tokens": n_new,
                     "reached_cap": n_new >= budget, "has_answer": ans is not None,
                     "answer": ans, "gold": it["gold"],
                     "correct": bool(check_answer(ans, it["gold"], it["benchmark"]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["r1-1.5b"])
    ap.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # Batch size is memory-bound, not compute-bound, on a 16 GB card: measured 113 tok/s at
    # batch 16 (8.2 GB reserved) vs 4.5 tok/s at batch 30 (28 GB reserved -> sysmem spill).
    # Measured reserved memory on this 16 GB card. NOTE it grows with sequence LENGTH too, not just
    # batch*budget "slots" -- at an identical 32768 slots: budget 4096 -> 9.3 GB (ok) but
    # budget 8192 -> 15.5 GB (SPILL). Observations:
    #   (bud 1024, b30) 30720 slots ->  5.9 GB ok
    #   (bud 4096, b8)  32768 slots ->  9.3 GB ok
    #   (bud 4096, b16) 65536 slots -> 24.6 GB SPILL
    #   (bud 8192, b4)  32768 slots -> 15.5 GB SPILL
    # MEASURED throughput at budget 8192: batch 4 (15.5 GB) = 45-96 tok/s vs batch 2 (8.4 GB) =
    # 39-50 tok/s. 15.5 GB still FITS in 16.3 GB, so it was not really spilling -- the old 15 GB
    # alarm was mis-calibrated. Use 32768 slots up to budget 8192, then back off for 16k+ where
    # the same slot count would genuinely exceed VRAM.
    ap.add_argument("--kv-cap", type=int, default=32768, help="max batch*budget token slots (memory guard)")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample each benchmark to N items (0 = all). Deterministic: a fixed "
                         "shuffle, so the same subset is reused across models/budgets/seeds.")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, _on_sigint)
    keep_awake(True)
    preflight_gpu()

    items_by_bench = {}
    for b in args.benchmarks:
        its = datamod.load_benchmark(b)
        if args.limit and len(its) > args.limit:
            import random as _r
            _r.Random(12345).shuffle(its)          # fixed subset across all cells
            its = sorted(its[:args.limit], key=lambda x: x["item_id"])
            print(f"[subsample] {b}: {args.limit} of {len(datamod.load_benchmark(b))} items", flush=True)
        items_by_bench[b] = its
    loaded_key, model, tok = None, None, None

    # BUDGET-OUTER: finish the cheap budgets across ALL models first (rank-instability headline is
    # at low/mid budgets and cheap), then work up to the expensive 16k/32k cells.
    for budget in sorted(args.budgets):
        # back off the slot cap for very long budgets, where memory grows with sequence length
        eff_cap = args.kv_cap if budget <= 8192 else args.kv_cap // 2
        bs = max(1, min(args.max_batch, eff_cap // budget))
        for mkey in args.models:
            for bench in args.benchmarks:
                items = items_by_bench[bench]
                for seed in args.seeds:
                    legacy, cell_dir = cell_paths(mkey, bench, budget, seed)
                    if legacy.exists() or (cell_dir / "_DONE").exists():
                        continue
                    done = done_item_ids(legacy, cell_dir)
                    todo = [it for it in items if it["item_id"] not in done]
                    if not todo:
                        cell_dir.mkdir(parents=True, exist_ok=True); (cell_dir / "_DONE").touch(); continue
                    if loaded_key != mkey:
                        if model is not None:
                            del model; model = None; gc.collect(); torch.cuda.empty_cache()
                        print(f"loading {mkey} ...", flush=True)
                        model, tok = load(mkey, args.device); loaded_key = mkey
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    todo.sort(key=lambda it: len(it["question"]))
                    part_n = len(list(cell_dir.glob("part_*.parquet")))
                    print(f"{mkey}/{bench}/b{budget}_s{seed}: {len(todo)} items, batch={bs}", flush=True)
                    i = 0
                    while i < len(todo):
                        chunk = todo[i:i + bs]
                        t0 = time.time()
                        try:
                            rows = gen_batch(model, tok, chunk, budget, seed, args.device)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if bs == 1:
                                raise
                            bs = max(1, bs // 2); print(f"      [OOM] batch->{bs}", flush=True); continue
                        df = pd.DataFrame(rows); df["model"] = mkey
                        df.to_parquet(cell_dir / f"part_{part_n:04d}.parquet"); part_n += 1
                        tot = int(df.n_new_tokens.sum()); dt = time.time() - t0
                        # Release cached blocks every batch: reserved memory otherwise ratchets up
                        # across batches until it exceeds VRAM and spills to system RAM.
                        resv = torch.cuda.max_memory_reserved() / 1e9
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        rate = tot / max(dt, 1e-6)
                        # 16.3 GB card: 15.5 GB still fits (measured faster than halving the batch),
                        # so only warn once we are genuinely at the ceiling.
                        warn = "  [!] SPILL? reserved>VRAM" if resv > 15.9 else ""
                        print(f"    batch {i//bs}: n={len(df)} acc={df.correct.mean():.2f} "
                              f"cap={df.reached_cap.mean():.2f} {tot}tok {dt:.0f}s "
                              f"({rate:.0f} tok/s) resv={resv:.1f}GB{warn}", flush=True)
                        i += len(chunk)
                        if STOP:
                            print("paused. re-run to resume.", flush=True)
                            if model is not None:
                                del model
                            gc.collect(); torch.cuda.empty_cache(); keep_awake(False); sys.exit(0)
                    (cell_dir / "_DONE").touch()
    if model is not None:
        del model; gc.collect(); torch.cuda.empty_cache()
    keep_awake(False)
    print("ALL DONE for requested grid.", flush=True)


if __name__ == "__main__":
    main()

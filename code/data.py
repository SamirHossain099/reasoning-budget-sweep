"""Benchmark loaders. Returns a uniform list of items: {item_id, question, gold, benchmark}."""
from __future__ import annotations

ANSWER_INSTR = " Please reason step by step, and put your final answer within \\boxed{}."

SOURCES = {
    "aime24": ("Maxwell-Jia/AIME_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}


def load_benchmark(name: str) -> list[dict]:
    from datasets import load_dataset
    repo, split = SOURCES[name]
    ds = load_dataset(repo, split=split)
    items = []
    for i, r in enumerate(ds):
        if name == "aime24":
            q, gold, iid = r["Problem"], str(r["Answer"]).strip(), str(r["ID"])
        elif name == "aime25":
            q, gold, iid = r["problem"], str(r["answer"]).strip(), str(r["id"])
        elif name == "math500":
            q, gold, iid = r["problem"], str(r["answer"]).strip(), str(r["unique_id"])
        else:
            raise ValueError(name)
        items.append({"item_id": iid, "question": q + ANSWER_INSTR, "gold": gold, "benchmark": name})
    return items

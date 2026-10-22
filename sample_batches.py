"""Sample passages (seed 42) and cut them into batches of 100 for the question writers.

Usage: python3 sample_batches.py          writes batches/<lang>_NN.jsonl {pid, title, section, text}
en 1000, fr/de/es 300 each (es has only 527 passages in total: Stripe translated about 120 pages). The whole language corpus is still
the search space at scoring time, the sample only decides which passages get a question.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "batches"
OUT.mkdir(exist_ok=True)
N = {"en": 1000, "fr": 300, "de": 300, "es": 300}
BATCH = 100

for lang, n in N.items():
    f = HERE / ("passages.jsonl" if lang == "en" else f"passages_{lang}.jsonl")
    rows = [json.loads(x) for x in open(f, encoding="utf-8")]
    random.Random(42).shuffle(rows)
    rows = rows[:n]
    for i in range(0, len(rows), BATCH):
        with open(OUT / f"{lang}_{i // BATCH:02d}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows[i : i + BATCH]:
                fh.write(json.dumps({k: r[k] for k in ("pid", "title", "section", "text")}, ensure_ascii=False) + "\n")
    print(lang, len(rows), "passages", -(-len(rows) // BATCH), "batches")

"""Ask the Stripe docs a question and see the top passages. Local models only, no API calls.

Usage:
  ./ask "How do I refund a payment?" [--lang en|fr|de|es] [--model octen|e5instruct] [--top 3]

Passage vectors come from cache/stripe_<lang>_<model>.npz, written by score.py, so an answer takes a few seconds
once the model is loaded. Only the question is embedded.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--lang", default="en", choices=["en", "fr", "de", "es"])
ap.add_argument("--model", default="octen", choices=["octen", "e5instruct"])
ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


if a.model == "octen":
    import octen_mlx  # 8-bit MLX conversion, no instruction

    q = norm(octen_mlx.embed([a.question]))
else:
    from sentence_transformers import SentenceTransformer

    from models import REVISIONS

    name = "intfloat/multilingual-e5-large-instruct"
    task = "Given a question about Stripe documentation, retrieve the passage that answers it"
    q = norm(SentenceTransformer(name, revision=REVISIONS[name]).encode([f"Instruct: {task}\nQuery: {a.question}"]))

with open(HERE / ("passages.jsonl" if a.lang == "en" else f"passages_{a.lang}.jsonl"), encoding="utf-8") as fh:
    corpus = [json.loads(x) for x in fh]
cache = np.load(HERE / "cache" / f"stripe_{a.lang}_{a.model}.npz")
vecs = cache["a"]
h = hashlib.sha256()
for r in corpus:
    h.update(f"{r['title']}\n{r['text']}".encode() + b"\0")
if len(vecs) != len(corpus) or ("hp" in cache and str(cache["hp"]) != h.hexdigest()[:16]):
    sys.exit("The cached vectors were built from other passages than passages*.jsonl. Rebuild them with score.py.")
sims = (q @ vecs.T)[0]

print(f"\n{a.question}\n({a.model}, {a.lang}, {len(corpus)} passages)\n")
for rank, i in enumerate(np.argsort(-sims)[: a.top], 1):
    p = corpus[i]
    body = " ".join(p["text"].split())
    print(f"{rank}. {sims[i]:.3f}  {p['title']}: {p['section']}\n   {p['url']}\n   {body[:220]}{'...' if len(body) > 220 else ''}\n")

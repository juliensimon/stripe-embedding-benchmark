"""Score embedding models on the Stripe docs test set.

Usage:
  uv run python score.py en [fr de es] [--call-openai]

Search space: every passage of that language (passages[_lang].jsonl). Queries: questions/<lang>_*.jsonl, skips dropped.
A question is matched to its passage by URL, section and a hash of the text, so a passage that changed since the questions
were written is dropped and counted, not silently mis-scored. A hit at k means the source passage is among the k nearest.
OpenAI embeddings are read from cache/. They are only fetched with --call-openai, which costs money and reads
OPENAI_API_KEY, OPENAI_BASE_URL and EMBEDDING_MODEL from the environment. Without the cache or the flag the OpenAI rows
are skipped. The 1024 row uses the real dimensions=1024 call when its cache exists (stripe_<lang>_openai1024.npz) and
otherwise cuts and renormalizes the 3072 vectors. All other embeddings are cached in cache/ too, so a rerun is free.
"""

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import octen_mlx  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
LANGS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["en"]
CALL_OPENAI = "--call-openai" in sys.argv
TASK = "Given a question about Stripe documentation, retrieve the passage that answers it"


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sha(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def st(name, query=lambda t: t, passage=lambda t: t, **kw):
    """A sentence-transformers model. query and passage rewrite the text before encoding (instruction or prefix)."""

    def run(q, p):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name, **kw)
        return norm(m.encode([query(t) for t in q], batch_size=16)), norm(m.encode([passage(t) for t in p], batch_size=16))

    return run


def octen(max_length):
    return lambda q, p: (norm(octen_mlx.embed(q, max_length)), norm(octen_mlx.embed(p, max_length, batch=4 if max_length > 1024 else 8)))


def openai_large(dimensions=None):
    """text-embedding-3-large through any OpenAI-compatible endpoint. Reads OPENAI_API_KEY, OPENAI_BASE_URL and
    EMBEDDING_MODEL (default text-embedding-3-large) from the environment. Costs money: about $0.13 per million tokens."""

    def run(q, p):
        import os

        from openai import OpenAI

        c = OpenAI()
        model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
        extra = {"dimensions": dimensions} if dimensions else {}

        def emb(texts):
            out = []
            for i in range(0, len(texts), 64):
                r = c.embeddings.create(model=model, input=texts[i : i + 64], **extra)
                out += [d.embedding for d in r.data]
            return norm(out)

        return emb(q), emb(p)

    return run


LEFT = {"tokenizer_kwargs": {"padding_side": "left"}}
QWEN_INSTRUCT = lambda t: f"Instruct: {TASK}\nQuery:{t}"  # noqa: E731  the format on the Qwen3-Embedding model card
E5_INSTRUCT = lambda t: f"Instruct: {TASK}\nQuery: {t}"  # noqa: E731  the format on the multilingual-e5-large-instruct card

# label: (cache key, function, vector size cut to). Without a cache of the real dimensions=1024 call, the 1024 row cuts and renormalizes the 3072 vectors.
MODELS = {
    "multilingual-e5-large-instruct (instruction on questions)": ("e5instruct", st("intfloat/multilingual-e5-large-instruct", query=E5_INSTRUCT), None),
    "multilingual-e5-large (query:/passage: prefixes)": ("e5large", st("intfloat/multilingual-e5-large", query=lambda t: f"query: {t}", passage=lambda t: f"passage: {t}"), None),
    "Qwen3-Embedding-0.6B base, no instruction": ("qwen3", st("Qwen/Qwen3-Embedding-0.6B", **LEFT), None),
    "Qwen3-Embedding-0.6B base, with instruction": ("qwen3i", st("Qwen/Qwen3-Embedding-0.6B", query=QWEN_INSTRUCT, **LEFT), None),
    "Octen-0.6B original (torch, 32k context)": ("octenst", st("Octen/Octen-Embedding-0.6B", **LEFT), None),
    "Octen-0.6B 8-bit MLX, 1,024 tokens": ("octen", octen(1024), None),
    "Octen-0.6B 8-bit MLX, 512 tokens": ("octen512", octen(512), None),
    "Octen-0.6B 8-bit MLX, 8,192 tokens": ("octenlong", octen(8192), None),
    "OpenAI 3-large at 1024": ("openai1024", openai_large(1024), None),
    "OpenAI 3-large at 3072": ("openai", openai_large(), None),
}
REF = "Octen-0.6B 8-bit MLX, 1,024 tokens"
PAID = "OpenAI 3-large at 3072"


def cached(lang, key, fn, q, p):
    """Return (query vectors, passage vectors). The cache stores a hash of the texts and refuses vectors for other texts."""
    f = CACHE / f"stripe_{lang}_{key}.npz"
    hq, hp = sha(q), sha(p)
    if f.exists():
        d = np.load(f)
        if "hq" in d and (str(d["hq"]) != hq or str(d["hp"]) != hp):
            sys.exit(f"{f.name} was built from other texts than the current passages and questions: delete it to rebuild")
        if d["q"].shape[0] == len(q) and d["a"].shape[0] == len(p):
            if "hq" not in d:  # first run with hashes: stamp a cache made before this check existed
                np.savez(f, q=d["q"], a=d["a"], hq=hq, hp=hp)
            return d["q"], d["a"]
    elif key == "openai1024" and (CACHE / f"stripe_{lang}_openai.npz").exists() and not CALL_OPENAI:
        d = np.load(CACHE / f"stripe_{lang}_openai.npz")  # no real dimensions=1024 call yet: cut the 3072 vectors
        return norm(d["q"][:, :1024]), norm(d["a"][:, :1024])
    elif key.startswith("openai") and not CALL_OPENAI:
        return None, None
    qv, pv = fn(q, p)
    np.savez(f, q=qv, a=pv, hq=hq, hp=hp)
    return qv, pv


for lang in LANGS:
    corpus = [json.loads(x) for x in open(HERE / ("passages.jsonl" if lang == "en" else f"passages_{lang}.jsonl"), encoding="utf-8")]
    where = {(r["url"], r["section"], hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16]): i for i, r in enumerate(corpus)}
    qs, dropped = [], 0
    for f in sorted(glob.glob(str(HERE / "questions" / f"{lang}_*.jsonl"))):
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r.get("skip"):
                continue
            if (r["url"], r["section"], r["sha"]) in where:
                qs.append(r)
            else:
                dropped += 1
    gold = np.array([where[(r["url"], r["section"], r["sha"])] for r in qs])
    # sibling chunks: passages cut from the same section of the same page (a long section becomes several chunks)
    sect = {}
    for i, r in enumerate(corpus):
        sect.setdefault((r["url"], r["section"]), []).append(i)
    sibling_sets = [np.array(sect[(corpus[g]["url"], corpus[g]["section"])]) for g in gold]
    has_sibling = np.array([len(s) > 1 for s in sibling_sets])
    words = np.array([len(corpus[g]["text"].split()) for g in gold])
    short = words <= 350

    def grams(t, n=4):
        w = re.findall(r"\w+", t.lower())
        return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

    # a question "overlaps" when it reuses 4 consecutive words of its passage: lexical help any model can use
    clean = np.array([not (grams(r["question"]) & grams(corpus[g]["text"])) for r, g in zip(qs, gold)])
    q_texts = [r["question"] for r in qs]
    p_texts = [f"{r['title']}\n{r['text']}" for r in corpus]
    print(f"\n=== {lang}: {len(qs)} questions ({dropped} dropped: passage changed or gone), {len(corpus)} passages ===")
    print(f"{has_sibling.mean():.0%} of gold passages have a sibling chunk from the same section; {short.mean():.0%} have 350 words or fewer; {clean.sum()} questions share no 4-word run with their passage")
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(qs), size=(1000, len(qs)))
    res, sec1 = {}, {}
    for label, (key, fn, cut) in MODELS.items():
        q, p = cached(lang, key, fn, q_texts, p_texts)
        if q is None:
            print(f"{label}: skipped (no cached OpenAI vectors; --call-openai fetches them and costs money)")
            continue
        if cut:
            q, p = norm(q[:, :cut]), norm(p[:, :cut])
        sims = q @ p.T
        order = (-sims).argsort(1)
        pos = (order == gold[:, None]).argmax(1)
        res[label] = pos
        sec1[label] = np.array([order[k, 0] in sibling_sets[k] for k in range(len(qs))])
        print(
            f"{label:58s} top1 {np.mean(pos == 0):6.1%}  top5 {np.mean(pos < 5):6.1%}  top10 {np.mean(pos < 10):6.1%}  dim {q.shape[1]}"
            f"  | same section top1 {sec1[label].mean():6.1%}"
            f"  | gold <=350 words top1 {np.mean(pos[short] == 0):6.1%}, longer {np.mean(pos[~short] == 0):6.1%}"
            f"  | no-overlap (n={clean.sum()}) top1 {np.mean(pos[clean] == 0):6.1%} top5 {np.mean(pos[clean] < 5):6.1%}"
        )
    print("bootstrap: the questions are redrawn at random with repeats, 1,000 times; gap (A minus B) on top-1 and top-5, 95% interval")
    print("primary comparisons: every row against Octen 1,024 and against OpenAI 3072. Every other difference is exploratory.")
    for ref in (REF, PAID):
        if ref not in res:
            continue
        for name in res:
            if name == ref or (ref == PAID and name == REF):
                continue
            line = f"  {name} vs {ref}:"
            for k in (1, 5):
                a, b = (res[name] < k).astype(int), (res[ref] < k).astype(int)
                d = (a[boot] - b[boot]).mean(1)
                line += f"  top{k} [{np.percentile(d, 2.5):+.1%}, {np.percentile(d, 97.5):+.1%}]"
            print(line)
    if REF in res and PAID in res:
        line = f"  {REF} vs {PAID}:"
        for k in (1, 5):
            d = ((res[REF] < k).astype(int)[boot] - (res[PAID] < k).astype(int)[boot]).mean(1)
            line += f"  top{k} [{np.percentile(d, 2.5):+.1%}, {np.percentile(d, 97.5):+.1%}]"
        print(line)
    json.dump({n: res[n].tolist() for n in res}, open(HERE / f"ranks_{lang}.json", "w"))

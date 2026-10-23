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

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import octen_mlx  # noqa: E402
from models import REVISIONS  # noqa: E402

CACHE = HERE / "cache"
CALL_OPENAI = False  # set by main(); the only switch that allows a paid call
TASK = "Given a question about Stripe documentation, retrieve the passage that answers it"


COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ  # colors only on a terminal, never in a redirected file


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def best(values, fmt, rows):
    """Format each value with fmt and paint the highest one among the eligible rows bold green."""
    top = max(v for v, ok in zip(values, rows, strict=True) if ok)
    return [paint(fmt(v), "1;32") if ok and v == top else fmt(v) for v, ok in zip(values, rows, strict=True)]


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

        m = SentenceTransformer(name, revision=REVISIONS[name], **kw)
        return norm(m.encode([query(t) for t in q], batch_size=16)), norm(m.encode([passage(t) for t in p], batch_size=16))

    return run


def octen(max_length):
    return lambda q, p: (norm(octen_mlx.embed(q, max_length)), norm(octen_mlx.embed(p, max_length, batch=4 if max_length > 1024 else 8)))


def openai_large(dimensions=None):
    """text-embedding-3-large through any OpenAI-compatible endpoint. Reads OPENAI_API_KEY, OPENAI_BASE_URL and
    EMBEDDING_MODEL (default text-embedding-3-large) from the environment. Costs money: about $0.13 per million tokens."""

    def run(q, p):
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

# label: (cache key, function). Without a cache of the real dimensions=1024 call, the 1024 row cuts and renormalizes the 3072 vectors.
MODELS = {
    "multilingual-e5-large-instruct (instruction on questions)": (
        "e5instruct",
        st("intfloat/multilingual-e5-large-instruct", query=E5_INSTRUCT),
    ),
    "multilingual-e5-large (query:/passage: prefixes)": (
        "e5large",
        st("intfloat/multilingual-e5-large", query=lambda t: f"query: {t}", passage=lambda t: f"passage: {t}"),
    ),
    "Qwen3-Embedding-0.6B base, no instruction": ("qwen3", st("Qwen/Qwen3-Embedding-0.6B", **LEFT)),
    "Qwen3-Embedding-0.6B base, with instruction": ("qwen3i", st("Qwen/Qwen3-Embedding-0.6B", query=QWEN_INSTRUCT, **LEFT)),
    "Octen-0.6B original (torch, 32k context)": ("octenst", st("Octen/Octen-Embedding-0.6B", **LEFT)),
    "Octen-0.6B 8-bit MLX, 1,024 tokens": ("octen", octen(1024)),
    "Octen-0.6B 8-bit MLX, 512 tokens": ("octen512", octen(512)),
    "Octen-0.6B 8-bit MLX, 8,192 tokens": ("octenlong", octen(8192)),
    "OpenAI 3-large at 1024": ("openai1024", openai_large(1024)),
    "OpenAI 3-large at 3072": ("openai", openai_large()),
}
SHORT = {  # short row names for the printed tables (the labels above stay the keys of ranks_*.json)
    "e5instruct": "e5-large-instruct",
    "e5large": "e5-large",
    "qwen3": "Qwen3-0.6B, no instruction",
    "qwen3i": "Qwen3-0.6B, instruction",
    "octenst": "Octen original (torch)",
    "octen": "Octen 8-bit, 1,024 tok",
    "octen512": "Octen 8-bit, 512 tok",
    "octenlong": "Octen 8-bit, 8,192 tok",
    "openai1024": "OpenAI 1024 dims",
    "openai": "OpenAI 3072 dims",
}
ABLATION = {"octen512", "octenlong"}  # context-length checks: printed, never highlighted
REF = "Octen-0.6B 8-bit MLX, 1,024 tokens"
PAID = "OpenAI 3-large at 3072"


def save_npz(f, **arrays):
    """Write to a temporary file and rename, so an interrupted run never leaves a corrupt cache."""
    tmp = f.with_name(f.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, f)


def cached(lang, key, fn, q, p):
    """Return (query vectors, passage vectors). The cache stores a hash of the texts and refuses vectors for other texts.
    A key that starts with "openai" reaches fn, the paid call, only when --call-openai was given."""
    f = CACHE / f"stripe_{lang}_{key}.npz"
    hq, hp = sha(q), sha(p)
    if f.exists():
        d = np.load(f)
        if "hq" not in d or str(d["hq"]) != hq or str(d["hp"]) != hp:
            sys.exit(f"{f.name} was built from other texts than the current passages and questions: delete it to rebuild")
        return d["q"], d["a"]
    if key.startswith("openai"):
        base = CACHE / f"stripe_{lang}_openai.npz"
        if key == "openai1024" and base.exists():  # no real dimensions=1024 call yet: cut the 3072 vectors
            d = np.load(base)
            if "hq" not in d or str(d["hq"]) != hq or str(d["hp"]) != hp:
                sys.exit(f"{base.name} was built from other texts than the current passages and questions: delete it to rebuild")
            return norm(d["q"][:, :1024]), norm(d["a"][:, :1024])
        if not CALL_OPENAI:
            return None, None
    qv, pv = fn(q, p)
    save_npz(f, q=qv, a=pv, hq=hq, hp=hp)
    return qv, pv


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def score_language(lang):
    corpus = read_jsonl(HERE / ("passages.jsonl" if lang == "en" else f"passages_{lang}.jsonl"))
    where = {(r["url"], r["section"], hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16]): i for i, r in enumerate(corpus)}
    qs, dropped = [], 0
    for f in sorted(glob.glob(str(HERE / "questions" / f"{lang}_*.jsonl"))):
        for r in read_jsonl(f):
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
    clean = np.array([not (grams(r["question"]) & grams(corpus[g]["text"])) for r, g in zip(qs, gold, strict=True)])
    q_texts = [r["question"] for r in qs]
    p_texts = [f"{r['title']}\n{r['text']}" for r in corpus]
    print(f"\n=== {lang}: {len(qs)} questions ({dropped} dropped: passage changed or gone), {len(corpus)} passages ===")
    print(
        f"{has_sibling.mean():.0%} of gold passages have a sibling chunk from the same section; "
        f"{short.mean():.0%} have 350 words or fewer; {clean.sum()} questions share no 4-word run with their passage"
    )
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(qs), size=(1000, len(qs)))
    res, sec1, dims = {}, {}, {}
    for label, (key, fn) in MODELS.items():
        q, p = cached(lang, key, fn, q_texts, p_texts)
        if q is None:
            print(f"{label}: skipped (no cached OpenAI vectors; --call-openai fetches them and costs money)")
            continue
        order = (-(q @ p.T)).argsort(1)
        pos = (order == gold[:, None]).argmax(1)
        res[label], dims[label] = pos, q.shape[1]
        sec1[label] = np.array([order[k, 0] in sibling_sets[k] for k in range(len(qs))])
    w = max(len(SHORT[MODELS[n][0]]) for n in res)
    name = lambda n: SHORT[MODELS[n][0]]  # noqa: E731
    names = list(res)
    cols = {
        "hit": [[np.mean(res[n] < k) for n in names] for k in (1, 5, 10)],
        "type": [
            [sec1[n].mean() for n in names],
            [np.mean(res[n][short] == 0) for n in names],
            [np.mean(res[n][~short] == 0) for n in names],
            [np.mean(res[n][clean] == 0) for n in names],
        ],
    }
    fmt = lambda wd: lambda v: f"{v:{wd}.1%}"  # noqa: E731
    eligible = [MODELS[n][0] not in ABLATION for n in names]
    hit = [best(c, fmt(6), eligible) for c in cols["hit"]]
    typ = [best(c, fmt(w_), eligible) for c, w_ in zip(cols["type"], (12, 12, 7, 20), strict=True)]
    print(
        "\nHIT RATE: share of questions whose source passage is ranked in the first k"
        + (" (best in green; the 512 and 8,192-token rows are context checks, not highlighted)" if COLOR else "")
    )
    print(f"{'':{w}s}  {'top-1':>6s} {'top-5':>6s} {'top-10':>6s} {'dim':>5s}")
    for i, n in enumerate(names):
        print(f"{name(n):{w}s}  {hit[0][i]} {hit[1][i]} {hit[2][i]} {dims[n]:5d}")
    print("\nTOP-1 BY QUESTION TYPE")
    print(f"{'':{w}s}  {'same section':>12s} {'<=350 words':>12s} {'longer':>7s} {f'no overlap (n={clean.sum()})':>20s}")
    for i, n in enumerate(names):
        print(f"{name(n):{w}s}  {typ[0][i]} {typ[1][i]} {typ[2][i]} {typ[3][i]}")

    def gap(x, ref):
        """(point gap, text) for top-1 and top-5: A minus B in points, bootstrap interval, * when it excludes zero."""
        out = []
        for k in (1, 5):
            a, b = (res[x] < k).astype(int), (res[ref] < k).astype(int)
            d = (a[boot] - b[boot]).mean(1)
            lo, hi = np.percentile(d, 2.5) * 100, np.percentile(d, 97.5) * 100
            point = (a.mean() - b.mean()) * 100
            out.append((point, hi < 0, f"{point:+6.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}"))
        return out

    def gap_table(rows, skip=()):
        """Print the rows: the largest positive gap in each column is green, a gap that is behind with an interval below zero is red."""
        top = [max([r[1][i][0] for r in rows if r[0] not in skip] + [0]) for i in (0, 1)]
        for label, cells in rows:
            txt = []
            for i, (point, behind, cell) in enumerate(cells):
                cell = f"{cell:>22s}"
                txt.append(
                    paint(cell, "1;32") if point == top[i] and point > 0 and label not in skip else paint(cell, "31") if behind else cell
                )
            print(f"{label:{w}s}  {txt[0]}   {txt[1]}")

    print("\nGAPS (A minus B, in points; 95% interval from a paired bootstrap, 1,000 redraws of the questions)")
    print(
        "An interval that excludes zero is marked *."
        + (" Green: the largest gain (context-check rows excluded). Red: behind, interval below zero." if COLOR else "")
    )
    print("Primary comparisons: every row against Octen 1,024 and OpenAI 3072. Every other difference is exploratory.")
    for ref in (REF, PAID):
        if ref not in res:
            continue
        print(f"\nvs {name(ref)}")
        print(f"{'':{w}s}  {'top-1 gap':>22s}   {'top-5 gap':>22s}")
        gap_table(
            [(name(n), gap(n, ref)) for n in res if n != ref and not (ref == PAID and n == REF)],
            skip={name(n) for n in res if MODELS[n][0] in ABLATION},
        )
    if REF in res and PAID in res:
        print(f"\n{name(REF)} vs {name(PAID)}")
        print(f"{'':{w}s}  {'top-1 gap':>22s}   {'top-5 gap':>22s}")
        gap_table([("", gap(REF, PAID))])
    with open(HERE / f"ranks_{lang}.json", "w") as fh:
        json.dump({n: res[n].tolist() for n in res}, fh)


def main():
    global CALL_OPENAI
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("langs", nargs="*", default=["en"], choices=["en", "fr", "de", "es"], metavar="lang", help="en fr de es (default en)")
    ap.add_argument("--call-openai", action="store_true", help="fetch OpenAI embeddings that are not cached (costs money)")
    args = ap.parse_args()
    CALL_OPENAI = args.call_openai
    CACHE.mkdir(exist_ok=True)
    for lang in args.langs:
        score_language(lang)


if __name__ == "__main__":
    main()

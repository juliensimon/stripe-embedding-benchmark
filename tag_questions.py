"""Add url, section and a text hash to every question, so it finds its passage even after the Stripe docs change.

Usage: python3 tag_questions.py    (one-off: the question files carried only a position, pid, in the 2026-10-21 fetch)
Safe to rerun: it reads passages[_lang].jsonl of the fetch the questions were written against.
"""

import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

for lang in ("en", "fr", "de", "es"):
    corpus = {
        r["pid"]: r
        for r in map(json.loads, open(HERE / ("passages.jsonl" if lang == "en" else f"passages_{lang}.jsonl"), encoding="utf-8"))
    }
    for f in sorted(glob.glob(str(HERE / "questions" / f"{lang}_*.jsonl"))):
        rows = [json.loads(x) for x in open(f, encoding="utf-8")]
        for r in rows:
            p = corpus[r["pid"]]
            r["url"], r["section"], r["sha"] = p["url"], p["section"], hashlib.sha256(p["text"].encode("utf-8")).hexdigest()[:16]
        with open(f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f, len(rows))

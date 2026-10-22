"""Fetch the Stripe docs listed in https://docs.stripe.com/llms.txt as Markdown and cut them into passages.

Usage: python3 build_corpus.py [lang]   lang = en (default), fr, de, es, it. Downloads once into pages[_lang]/, rerun is free.
Stripe picks the locale from Accept-Language and falls back to French for languages it does not serve (nl).
Writes pages[_lang]/*.md and passages[_lang].jsonl {pid, url, title, section, text}. Both are gitignored copies of public docs:
re-fetch on recording day, do not redistribute.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
SUFFIX = "" if LANG == "en" else f"_{LANG}"
LOCALE = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "it": "it-IT"}[LANG]
PAGES = HERE / f"pages{SUFFIX}"
PAGES.mkdir(exist_ok=True)
INDEX = "https://docs.stripe.com/llms.txt"  # the English index lists the page URLs for every locale
MIN_WORDS, MAX_WORDS = 60, 450


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "julien-embedding-eval/1.0", "Accept-Language": f"{LOCALE},{LANG};q=0.9"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def slug(url):
    return re.sub(r"[^a-z0-9]+", "_", url.removeprefix("https://docs.stripe.com/").lower()).strip("_")


def fetch_pages():
    index = get(INDEX)
    urls = sorted(set(re.findall(r"\((https://docs\.stripe\.com/[^)\s]+?\.md)(?:#[^)]*)?\)", index)))
    urls = [u for u in urls if "/api/" not in u and not u.endswith("/api.md")]  # the API reference is generated, not prose
    for u in urls:
        f = PAGES / f"{slug(u)}.md"
        if f.exists():
            continue
        try:
            f.write_text(get(u), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("skip", u, e)
        time.sleep(0.2)
    return urls


def split(text):
    """Split on level-2 headings, then on blank lines when a section is too long."""
    title = (re.match(r"#\s+(.+)", text) or [None, ""])[1]
    parts = re.split(r"\n(?=## )", text)
    for part in parts:
        head = (re.match(r"##\s+(.+)", part) or [None, ""])[1]
        words = part.split()
        if len(words) < MIN_WORDS:
            continue
        if len(words) <= MAX_WORDS:
            yield title, head, part.strip()
            continue
        buf, n = [], 0
        for para in part.split("\n\n"):
            w = len(para.split())
            if n + w > MAX_WORDS and n >= MIN_WORDS:
                yield title, head, "\n\n".join(buf)
                buf, n = [], 0
            buf.append(para)
            n += w
        if n >= MIN_WORDS:
            yield title, head, "\n\n".join(buf)


STOP = {
    "en": {"the", "and", "to", "of", "a", "is", "you", "for", "in", "your", "can", "with", "that", "this", "are"},
    "fr": {"le", "la", "les", "de", "des", "et", "un", "une", "vous", "pour", "en", "votre", "que", "est", "dans", "du"},
    "de": {"der", "die", "das", "und", "zu", "ist", "sie", "für", "ihr", "ihre", "mit", "von", "den", "ein", "eine", "im"},
    "es": {"el", "la", "los", "las", "de", "y", "que", "en", "un", "una", "para", "su", "con", "es", "por", "del", "se", "a", "tu", "tus", "al", "lo", "como", "o", "puedes", "si", "no", "este", "esta"},
    "it": {"il", "la", "le", "di", "e", "che", "un", "una", "per", "in", "con", "è", "del", "della", "sono", "i"},
}[LANG]


def keep(text):
    """Drop reference blobs, passages not in the requested language and passages that are mostly code."""
    words = text.split()
    if len(words) > 600:
        return False
    prose = re.sub(r"```.*?```", " ", text, flags=re.S).split()
    if len(prose) < MIN_WORDS:
        return False
    return sum(w.lower().strip(".,()") in STOP for w in prose) / len(prose) >= 0.12


if __name__ == "__main__":
    urls = fetch_pages()
    out, k, seen = [], 0, set()
    for u in urls:
        f = PAGES / f"{slug(u)}.md"
        if not f.exists():
            continue
        for title, head, text in split(f.read_text(encoding="utf-8")):
            if not keep(text):
                continue
            key = re.sub(r"\W+", " ", text.lower())[:400]  # Stripe repeats shared sections on several pages: a duplicate has two right answers
            if key in seen:
                continue
            seen.add(key)
            out.append({"pid": f"p{k:05d}", "url": u, "title": title, "section": head, "text": text})
            k += 1
    with open(HERE / f"passages{SUFFIX}.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = sorted(len(r["text"].split()) for r in out)
    print(f"{len(urls)} pages listed, {len(list(PAGES.glob('*.md')))} fetched, {len(out)} passages, words median {w[len(w)//2]} max {w[-1]}")

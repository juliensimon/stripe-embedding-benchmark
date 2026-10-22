# Does paid embedding beat a free 0.6B model on the Stripe docs?

A retrieval test built from the public Stripe documentation in English, French, German and Spanish. It scores
OpenAI `text-embedding-3-large`, `multilingual-e5-large-instruct`, `multilingual-e5-large`, Qwen3-Embedding-0.6B and
Octen-Embedding-0.6B, and reports every gap with a 95% interval. A local demo, `./ask`, searches the docs on your machine.

Results of the run described in the video: `results.txt` (measured 2026-10-21 on the docs fetched that day).

## How it works

1. `build_corpus.py` fetches the pages listed in `https://docs.stripe.com/llms.txt` as Markdown, cuts them at second-level
   headings, and drops short, code-heavy, non-prose and duplicate pieces. The passages are public documentation owned by
   Stripe and are not in this repository: you fetch them.
2. `questions/` holds one question per sampled passage (en 1,000 sampled, fr, de and es 300 each, skips removed), written by
   Claude subagents with the brief in `questions/PROMPT.md`. The writer is not one of the models under test. Each question
   carries the URL, section and a hash of the passage it was written from.
3. `score.py` embeds questions and passages with every model, ranks the passages for each question, and prints top-1, top-5
   and top-10. A question counts as a hit when its source passage is among the first k. It also prints the share of top-1
   results that land in the right section (a long section is several chunks), and the split by passage length.
4. The gap between two models is a paired bootstrap over the questions: redraw the questions at random with repeats 1,000
   times, compute the gap each time, read the 2.5th and 97.5th percentiles. If the interval contains zero, the test cannot
   tell the two apart.

## Run it

Apple silicon Mac, [uv](https://docs.astral.sh/uv/).

```
uv run python build_corpus.py          # then: fr, de, es
uv run python score.py en fr de es
./ask "Which coupon setting can I still edit after creating the coupon?" --model octen
```

- The Stripe docs change. The question files match passages by URL, section and text hash, so a passage that changed since
  2026-10-21 is dropped and counted in the score output. The numbers in `results.txt` come from the 2026-10-21 fetch.
- `score.py` never calls a paid API by default. The OpenAI rows read `cache/` when it exists and are skipped otherwise;
  `--call-openai` fetches them and costs money (about $0.13 per million tokens, roughly 4 million tokens for the four
  languages). It reads `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `EMBEDDING_MODEL` from the environment. The 1024-dimension
  row uses the real `dimensions` parameter when its cache exists and otherwise cuts and renormalizes the 3072 vectors.
- Octen runs as the 8-bit MLX conversion `juliensimon/Octen-Embedding-0.6B-mlx-8bit` (a community conversion, Apache 2.0)
  and, for comparison, as the original `Octen/Octen-Embedding-0.6B` through sentence-transformers.

## What the test does not cover

Code and tables (code-heavy chunks are dropped), reranking, keyword or hybrid search, real user queries (the questions are
written from passages and are longer and tidier than search-box queries), and documents outside the Stripe docs.
Qwen3-Embedding was itself trained on questions written by a language model from passages, the recipe used here.

## License

Code: Apache 2.0. Questions and write-up: CC BY 4.0. The Stripe documentation is not included.

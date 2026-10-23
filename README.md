# Stripe docs embedding benchmark

**Does paying for OpenAI embeddings beat a free 0.6B model? A retrieval test built from Stripe's public documentation in English, French, German and Spanish.**

On this test, `text-embedding-3-large` shows no lead over the best free models (Octen-Embedding-0.6B and the plain multilingual-e5-large) in any of the four languages. It does lead the mismatched setups: in English over e5-instruct and over Qwen3 without its instruction, and in Spanish over instructed Qwen3 at top-5. Every gap below comes with a 95% interval. The repository holds the questions, the scoring code, the raw results and a local search demo. It accompanies a YouTube video by Julien Simon.

## Results

Top-1 / top-5: the share of questions whose source passage comes back first / among the first five. Measured 2026-10-21 and 2026-10-22 on the Stripe docs fetched 2026-10-21 (questions: en 863, fr 251, de 271, es 241). Full output with every interval: [`results.txt`](results.txt).

| Model | Max context (tokens) | en | fr | de | es |
|---|---|---|---|---|---|
| OpenAI text-embedding-3-large, 3072 dimensions | about 8,192 | 64.9 / 90.6 | 54.2 / 83.7 | 58.7 / 84.5 | 66.0 / 92.9 |
| OpenAI text-embedding-3-large, `dimensions=1024` | about 8,192 | 64.2 / 88.9 | 51.0 / 83.3 | 57.2 / 84.1 | 66.0 / 90.9 |
| multilingual-e5-large | 512 | 66.7 / 90.6 | 53.4 / 83.3 | 67.5 / 91.9 | 72.6 / 92.1 |
| multilingual-e5-large-instruct | 512 | 60.1 / 86.8 | 51.8 / 82.9 | 57.2 / 86.0 | 63.9 / 89.6 |
| Qwen3-Embedding-0.6B, no instruction | 32,768 | 57.8 / 85.5 | 54.6 / 81.3 | 60.1 / 85.2 | 67.6 / 91.7 |
| Qwen3-Embedding-0.6B, card instruction | 32,768 | 64.3 / 89.6 | 50.2 / 82.1 | 57.9 / 84.5 | 63.9 / 87.1 |
| Octen-Embedding-0.6B, original (torch) | 32,768 | 65.5 / 91.4 | 57.4 / 83.7 | 60.9 / 86.7 | 71.0 / 93.8 |
| Octen-Embedding-0.6B, 8-bit MLX (1,024-token cap) | 32,768 | 65.9 / 91.5 | 57.8 / 85.3 | 62.4 / 87.1 | 70.5 / 93.4 |
| Octen-Embedding-0.6B, 8-bit MLX (512-token cap) | 32,768 | 67.8 / 92.1 | 58.2 / 85.7 | 63.5 / 89.7 | 71.0 / 92.9 |
| Octen-Embedding-0.6B, 8-bit MLX (8,192-token cap) | 32,768 | 65.7 / 91.8 | 58.2 / 84.9 | 61.3 / 87.1 | 70.5 / 93.8 |

What the intervals support:

- **English:** Octen, OpenAI and the plain e5-large cannot be told apart (Octen minus OpenAI 3072, top-1: -2.3 to +4.5 points). The e5-instruct model trails Octen by 2.4 to 9.5 points.
- **Other languages:** these have about 250 questions each, so the test only rules out gaps larger than about 10 points. Within that, OpenAI shows no lead over Octen or the plain e5-large; the only French, German and Spanish lead it shows is over instructed Qwen3 in Spanish at top-5 (-10.0 to -1.7 for Qwen3). In German and Spanish the plain e5-large is ahead of OpenAI 3072 (+2.6 to +15.1 and +0.8 to +13.3 at top-1).
- **Instructions matter as much as the model:** the instruction on the Qwen3 card lifts the base model by 6.5 points in English and lowers it by about 4 points in French and Spanish.
- **Context length does not explain e5's gap:** Octen reading only 512 tokens scores 67.8 in English, higher than at 1,024.
- **The 8-bit conversion costs nothing measurable** against the original Octen checkpoint.

## Method

1. [`build_corpus.py`](build_corpus.py) fetches the pages listed in <https://docs.stripe.com/llms.txt> as Markdown, cuts them at second-level headings (sections over 450 words are split again on paragraphs), and drops passages that are short, mostly code, in another language or duplicated. The passages are Stripe's documentation and are not in this repository.
2. [`questions/`](questions) holds one question per sampled passage, written by Claude subagents with the brief in [`questions/PROMPT.md`](questions/PROMPT.md). The writer is not a model under test. Each question carries the URL, section and hash of its passage, so a passage that changed since the fetch is dropped and counted, never silently mis-scored.
3. [`score.py`](score.py) embeds questions and passages with each model and ranks the passages for every question. A hit at k means the source passage is among the k nearest.
4. The gap between two models is a paired bootstrap over the questions: redraw them at random with repeats 1,000 times, compute the gap each time, and read the 2.5th and 97.5th percentiles. If the interval contains zero, the test cannot tell the two models apart. Primary comparisons are every model against Octen and against OpenAI 3072; everything else is exploratory. That is about 70 unadjusted intervals across the four languages, so a single bound that clears zero by under a point (the Spanish e5-large over OpenAI 3072, +0.8) is weak evidence; the null results do not depend on this.

Checks run by script: the share of gold passages with a sibling chunk from the same section, the split by passage length, and a second score on the questions that share no four-word run with their passage.

## Run it

Apple silicon Mac and [uv](https://docs.astral.sh/uv/). Every model loads at the Hub revision in [`models.py`](models.py), the one the numbers were measured with.

```
uv run python build_corpus.py          # then: fr, de, es (writes pages*/ and passages*.jsonl)
uv run python score.py en fr de es
./ask "Which coupon setting can I still edit after creating the coupon?" --model octen
uv run pytest && uvx ruff check .     # tests and lint; no model or network call
```

- The Stripe docs change. The numbers above come from the 2026-10-21 fetch; a fresh fetch drops the questions whose passage changed, and the score output says how many.
- `score.py` never calls a paid API by default. The OpenAI rows read `cache/` when present and are skipped otherwise. `--call-openai` fetches them, costs money (about $0.13 per million tokens, roughly 4 million tokens for the four languages) and reads `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `EMBEDDING_MODEL` from the environment.
- My MLX conversions of Octen (all public, Apache 2.0, nine repos): [0.6B](https://huggingface.co/juliensimon/Octen-Embedding-0.6B-mlx-8bit), [4B](https://huggingface.co/juliensimon/Octen-Embedding-4B-mlx-8bit) and [8B](https://huggingface.co/juliensimon/Octen-Embedding-8B-mlx-8bit), each in bf16, 8bit and 4bit (replace `8bit` in the link). The test uses the 0.6B 8-bit one.
- Octen runs as the community 8-bit MLX conversion [`juliensimon/Octen-Embedding-0.6B-mlx-8bit`](https://huggingface.co/juliensimon/Octen-Embedding-0.6B-mlx-8bit) (Apache 2.0) of [`Octen/Octen-Embedding-0.6B`](https://huggingface.co/Octen/Octen-Embedding-0.6B), and for comparison as the original through sentence-transformers.
- `./ask` embeds one question locally and prints the closest passages from the cached vectors. No API call.

## What the test does not cover

Code and tables (code-heavy chunks are dropped), reranking, keyword or hybrid search, real user queries (the questions are written from passages and are longer and tidier than search-box queries), server-grade builds of the models, and documents other than the Stripe docs. Qwen3-Embedding was itself trained on questions that a language model wrote from passages, the recipe used here, so this style may suit it and Octen.

## Layout

| Path | What it is |
|---|---|
| `build_corpus.py`, `sample_batches.py`, `tag_questions.py` | Fetch and cut the docs, sample passages for the question writers, tag questions with their passage |
| `questions/` | The questions and the writer's brief |
| `score.py`, `ask.py`, `ask`, `octen_mlx.py`, `models.py` | Scoring, the local demo, its wrapper, the Octen loader, the pinned model revisions |
| `tests/` | Tests of the splitting, filtering, cache and paid-call guard |
| `results.txt` | Reference output of `score.py en fr de es` (`score.py` also writes per-question ranks to `ranks_<lang>.json`, which are not tracked) |

## License

Code: [Apache 2.0](LICENSE). Questions and brief: [CC BY 4.0](questions/LICENSE); the questions describe Stripe content in new words and quote no passage. The Stripe documentation is not included and remains Stripe's.

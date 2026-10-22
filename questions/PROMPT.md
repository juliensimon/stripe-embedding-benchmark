# Question-writing brief (same text for every batch)

You write test questions for an embedding-retrieval benchmark built on the Stripe documentation.

Input: a JSONL batch file. Each line has `pid`, `title`, `section`, `text` (one documentation passage).
Output: a JSONL file, one line per input passage: {"pid": "...", "question": "...", "skip": false}

Rules:
1. Write ONE question per passage, in the language of the passage (English, French, German or Spanish).
2. The question is what a developer or a business user would type in a search box or ask a colleague while working with Stripe. Natural, specific, One sentence, 8 to 25 words, asking one thing (no "and" joining two questions).
3. The passage must contain the answer, and it must be the ONLY passage in the whole Stripe docs that fully answers it. Anchor the question on a concrete detail of this passage: a parameter, a limit, a condition, a step, a status, a rule. Not on the topic in general.
4. Do NOT reuse 4 or more consecutive words from the passage, except product names and code identifiers (for example "Payment Element", `payment_intent`). Rephrase everything else with different words. Do NOT mention "the passage", "the documentation" or "this page". Do not name the page title.
5. No yes/no questions. No questions that need the reader to see the passage to make sense.
6. If a passage is a generic overview, a changelog, a list of links, mostly code with no explanation, or you cannot write a question that only this passage answers, set "skip": true and "question": "".
7. Do not use any tool except reading the batch file and writing the output file. Do not browse. Write the output file once, complete.

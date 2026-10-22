"""Octen-Embedding-0.6B on Apple silicon: the 8-bit MLX conversion, run with mlx-embeddings.

The conversion is juliensimon/Octen-Embedding-0.6B-mlx-8bit (Apache 2.0, a community conversion of Octen/Octen-Embedding-0.6B).
No instruction prefix: the model card documents none. Text is cut at max_length tokens, the vectors come back L2-normalized.
"""

import numpy as np

from models import REVISIONS

MODEL = "juliensimon/Octen-Embedding-0.6B-mlx-8bit"
_state = {}


def embed(texts, max_length=1024, batch=8):
    """Embed a list of texts and return a (len(texts), dim) float32 array, in the order given."""
    import mlx.core as mx
    from mlx_embeddings import load
    from mlx_embeddings.utils import get_model_path

    if "model" not in _state:
        _state["model"], _state["tok"] = load(str(get_model_path(MODEL, REVISIONS[MODEL])))
    model, tok = _state["model"], _state["tok"]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together: little padding
    out = [None] * len(texts)
    for i in range(0, len(order), batch):
        idx = order[i : i + batch]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_length, return_tensors="mlx")
        e = model(b["input_ids"], attention_mask=b["attention_mask"]).text_embeds
        mx.eval(e)
        for j, v in zip(idx, np.array(e.astype(mx.float32)), strict=True):
            out[j] = v
    return np.stack(out)

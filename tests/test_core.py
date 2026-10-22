"""Pure-function tests. Nothing here loads a model or calls a network API."""

import sys

import numpy as np
import pytest


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_corpus.py"])  # the module reads its language from argv at import
    sys.modules.pop("build_corpus", None)
    import build_corpus

    return build_corpus


@pytest.fixture
def score(monkeypatch, tmp_path):
    import score as s

    monkeypatch.setattr(s, "CACHE", tmp_path)
    monkeypatch.setattr(s, "CALL_OPENAI", False)
    return s


def words(n, w="word"):
    return " ".join([w] * n)


def test_split_keeps_a_section_under_the_limit_whole(corpus):
    text = f"# Title\n\n## Intro\n{words(100)}\n\n## Tiny\n{words(10)}"
    out = list(corpus.split(text))
    assert [(t, h) for t, h, _ in out] == [("Title", "Intro")]


def test_split_cuts_a_long_section_on_paragraphs(corpus):
    paras = "\n\n".join(words(200) for _ in range(5))
    out = list(corpus.split(f"# T\n\n## Long\n{paras}"))
    assert len(out) > 1
    assert all(len(t.split()) <= corpus.MAX_WORDS + 200 for _, _, t in out)


def test_keep_drops_code_and_other_languages(corpus):
    english = " ".join(["you can use the API for your account with this"] * 12)
    assert corpus.keep(english)
    assert not corpus.keep("```\n" + words(200) + "\n```")
    assert not corpus.keep(" ".join(["Sie können die Zahlung mit der Karte"] * 20))
    assert not corpus.keep(words(700))


def test_sha_depends_on_every_text_and_their_order(score):
    assert score.sha(["a", "b"]) == score.sha(["a", "b"])
    assert score.sha(["a", "b"]) != score.sha(["b", "a"])
    assert score.sha(["ab"]) != score.sha(["a", "b"])  # the separator matters


def test_norm_returns_unit_rows(score):
    x = score.norm([[3.0, 4.0], [0.0, 2.0]])
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


def test_cache_refuses_vectors_built_from_other_texts(score, tmp_path):
    q, p = ["q1"], ["p1"]
    score.save_npz(tmp_path / "stripe_en_e5large.npz", q=np.ones((1, 2)), a=np.ones((1, 2)), hq="x", hp="y")
    with pytest.raises(SystemExit):
        score.cached("en", "e5large", lambda *_: pytest.fail("must not recompute"), q, p)


def test_cache_without_hashes_is_refused_not_trusted(score, tmp_path):
    np.savez(tmp_path / "stripe_en_e5large.npz", q=np.ones((1, 2)), a=np.ones((1, 2)))
    with pytest.raises(SystemExit):
        score.cached("en", "e5large", lambda *_: pytest.fail("must not recompute"), ["q"], ["p"])


def test_openai_is_never_called_without_the_flag(score, tmp_path):
    called = []
    assert score.cached("en", "openai", lambda *_: called.append(1), ["q"], ["p"]) == (None, None)
    np.savez(tmp_path / "stripe_en_openai1024.npz", q=np.ones((2, 2)), a=np.ones((2, 2)))  # stale cache, wrong shape, no hashes
    with pytest.raises(SystemExit):
        score.cached("en", "openai1024", lambda *_: called.append(1), ["q"], ["p"])
    assert not called


def test_openai1024_cuts_the_3072_vectors_when_hashes_match(score, tmp_path):
    q, p = ["q"], ["p"]
    full = np.random.default_rng(0).normal(size=(1, 3072)).astype(np.float32)
    score.save_npz(tmp_path / "stripe_en_openai.npz", q=full, a=full, hq=score.sha(q), hp=score.sha(p))
    qv, _ = score.cached("en", "openai1024", lambda *_: pytest.fail("paid call"), q, p)
    assert qv.shape == (1, 1024) and np.isclose(np.linalg.norm(qv), 1.0)


def test_save_npz_leaves_no_temporary_file(score, tmp_path):
    score.save_npz(tmp_path / "x.npz", a=np.zeros(3))
    assert [f.name for f in tmp_path.iterdir()] == ["x.npz"]

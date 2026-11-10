"""
controls_synthetic.py: null-model and synthetic corpus generators for the control runs.

Every text-transform control has the signature:

    fn(texts: list[str], seed: int) -> list[str]     # same length as `texts`

`texts` is the stemmed, whitespace-joined corpus that load_corpus() returns (one string
per paper). Each function is pure and fully determined by `seed`, so runs reproduce exactly.

The point of these is to destroy the *real* topic structure in a controlled way, so that
when we run LDA/HDP on them we can see what "no real topics" looks like and compare it to
the real corpus. Which kind of structure each one destroys:

  shuffle   -- same words corpus-wide, positions shuffled across papers; destroys which
               words co-occur *within* a paper (keeps each paper's length + global counts).
  resample  -- each paper re-drawn from the corpus-wide word distribution ("same words,
               different counts"); destroys per-paper word identity, keeps global frequencies.
  uniform   -- every token replaced by a word drawn UNIFORMLY from the vocabulary (not
               weighted by frequency); destroys frequency structure too.
  gibberish -- random non-vocabulary tokens interleaved among the real words (adds noise).
  nowm      -- "working memory" removed (its stems dropped) and then modelled as usual.

make_synthetic_known_topics() is different: a POSITIVE control that builds a corpus with
*known* planted topics, so we can check the models actually recover real structure when it
is there.

Vocabulary note (handled by controls.py, not here): shuffle / resample / uniform / nowm are
scored against the REAL corpus's frozen vocabulary; gibberish and synthetic introduce new
words, so they are fit on their own corpus. See FRESH_VOCAB_CONTROLS below.
"""
import numpy as np

_ALPHA = list("abcdefghijklmnopqrstuvwxyz")

# Stems of "working memory" (corpus is already Snowball-stemmed: working->work, memory->memori).
WM_STEMS = {"work", "memori", "memor", "workingmemori"}

# Controls that introduce words NOT in the real vocabulary, so they must be vectorized on
# their own corpus rather than the frozen real vocabulary.
FRESH_VOCAB_CONTROLS = {"gibberish", "synthetic"}


def _docs(texts):
    return [t.split() for t in texts]


def shuffle_corpus_words(texts, seed):
    """Pool every token in the corpus, shuffle, and redistribute keeping each paper's length."""
    rng = np.random.default_rng(seed)
    docs = _docs(texts)
    lengths = [len(d) for d in docs]
    pool = np.array([w for d in docs for w in d], dtype=object)
    rng.shuffle(pool)
    out, i = [], 0
    for L in lengths:
        out.append(" ".join(pool[i:i + L].tolist()))
        i += L
    return out


def resample_words_global(texts, seed):
    """Keep each paper's length; redraw its words from the corpus-wide word distribution
    (with replacement, weighted by global frequency). 'Same words, different counts.'"""
    rng = np.random.default_rng(seed)
    docs = _docs(texts)
    pool = np.array([w for d in docs for w in d], dtype=object)
    if len(pool) == 0:
        return list(texts)
    vocab, counts = np.unique(pool, return_counts=True)
    probs = counts / counts.sum()
    out = []
    for d in docs:
        if not d:
            out.append("")
            continue
        draw = rng.choice(len(vocab), size=len(d), replace=True, p=probs)
        out.append(" ".join(vocab[draw].tolist()))
    return out


def replace_uniform_vocab(texts, seed):
    """Replace every token with a word drawn UNIFORMLY from the vocabulary (flat, not
    weighted by how common real words are). Keeps each paper's length."""
    rng = np.random.default_rng(seed)
    docs = _docs(texts)
    vocab = np.array(sorted({w for d in docs for w in d}), dtype=object)
    if len(vocab) == 0:
        return list(texts)
    out = []
    for d in docs:
        if not d:
            out.append("")
            continue
        draw = rng.integers(0, len(vocab), size=len(d))
        out.append(" ".join(vocab[draw].tolist()))
    return out


def insert_gibberish(texts, seed, ratio=1.0, pool_size=None):
    """Interleave random gibberish (non-vocabulary) tokens among the real words.
    `ratio` = gibberish tokens added per real token. A fixed gibberish pool is reused so the
    noise words recur (otherwise min_df would prune every one of them)."""
    rng = np.random.default_rng(seed)
    real_vocab = {w for t in texts for w in t.split()}

    def mkword():
        return "".join(rng.choice(_ALPHA, size=int(rng.integers(4, 9))))

    n_gib = pool_size or max(50, len(real_vocab) // 4)
    pool = []
    seen = set()
    while len(pool) < n_gib:
        g = mkword()
        if g not in real_vocab and g not in seen:
            pool.append(g); seen.add(g)
    pool = np.array(pool, dtype=object)

    out = []
    for t in texts:
        words = t.split()
        n_insert = int(len(words) * ratio)
        combined = list(words)
        if n_insert:
            combined += pool[rng.integers(0, len(pool), size=n_insert)].tolist()
        rng.shuffle(combined)  # bag-of-words: position is irrelevant to LDA/HDP
        out.append(" ".join(combined))
    return out


def drop_working_memory(texts, seed=0):
    """Remove the 'working memory' stems from every paper, then model as usual.
    (seed unused; kept for a uniform control signature.)"""
    out = []
    for t in texts:
        out.append(" ".join(w for w in t.split() if w not in WM_STEMS))
    return out


def make_synthetic_known_topics(n_docs, n_topics=5, seed=0, words_per_topic=40,
                                doc_len=150, noise=0.1):
    """Positive control: build a corpus with KNOWN planted topics.

    Each topic owns a disjoint pool of `words_per_topic` invented (but alphabetic, so the
    vectorizer keeps them) words. Each document is assigned one dominant topic and filled
    mostly from that topic's pool plus a `noise` fraction drawn from the whole vocabulary.
    Returns (texts, ground_truth) where ground_truth records topic_words and doc_topics so a
    recovered model can be scored against the planted structure.
    """
    rng = np.random.default_rng(seed)

    def mkword():
        return "".join(rng.choice(_ALPHA, size=int(rng.integers(5, 8))))

    words, seen = [], set()
    while len(words) < n_topics * words_per_topic:
        w = mkword()
        if w not in seen:
            words.append(w); seen.add(w)
    rng.shuffle(words)
    groups = [words[k * words_per_topic:(k + 1) * words_per_topic] for k in range(n_topics)]
    all_words = np.array(words, dtype=object)

    # Peaky (Zipfian) within-topic word weights: a few words dominate each topic, like real
    # topics. A flat (uniform) within-topic distribution is unrealistic and gets rejected by
    # HDP's flat-topic filter, so a uniform positive control would understate HDP's ability.
    zipf = 1.0 / np.arange(1, words_per_topic + 1)
    zipf /= zipf.sum()

    texts, doc_topics = [], []
    n_noise = int(doc_len * noise)
    n_main = doc_len - n_noise
    for _ in range(n_docs):
        k = int(rng.integers(0, n_topics))
        doc_topics.append(k)
        main = rng.choice(groups[k], size=n_main, replace=True, p=zipf)
        noisew = rng.choice(all_words, size=n_noise, replace=True) if n_noise else np.array([], dtype=object)
        doc = np.concatenate([main, noisew])
        rng.shuffle(doc)
        texts.append(" ".join(doc.tolist()))

    ground_truth = {
        "n_topics": n_topics, "words_per_topic": words_per_topic,
        "doc_len": doc_len, "noise": noise, "seed": seed,
        "topic_words": {str(k): groups[k] for k in range(n_topics)},
        "doc_topics": doc_topics,
    }
    return texts, ground_truth


# name -> text-transform function (synthetic is handled separately, it is generative)
TEXT_CONTROLS = {
    "shuffle": shuffle_corpus_words,
    "resample": resample_words_global,
    "uniform": replace_uniform_vocab,
    "gibberish": insert_gibberish,
    "nowm": drop_working_memory,
}

ALL_CONTROLS = ["real"] + list(TEXT_CONTROLS) + ["synthetic"]

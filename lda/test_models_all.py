"""
Test both LDA and HDP across 100, 1000, and optionally 10000 papers.
Reports topic words, diversity, overlap, coherence.
"""
import sys, os, zipfile, tempfile, time, math
import numpy as np
sys.path.insert(0, 'src')

from preprocessing import (
    preprocess_text, extract_text_from_pdf, clean_pdf_text,
    get_all_stopwords, prepare_gensim_corpus
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import HdpModel, CoherenceModel
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)


def load_corpus(zip_path):
    """Extract and preprocess PDFs from a ZIP."""
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmpdir)

    stopwords = get_all_stopwords()
    texts = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.endswith('.pdf') and not f.startswith('._'):
                t = extract_text_from_pdf(os.path.join(root, f))
                if t:
                    c = clean_pdf_text(t)
                    p = preprocess_text(c)
                    if p and len(p.split()) > 20:
                        texts.append(p)
    return texts, stopwords


def vectorize(texts, stopwords):
    """Vectorize with same params as app.py."""
    n = len(texts)
    if n < 50: mf = 1000
    elif n < 500: mf = 3000
    elif n < 5000: mf = 5000
    else: mf = 8000

    vec = CountVectorizer(
        max_df=0.35, min_df=max(2, n // 50),
        stop_words=stopwords,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        max_features=mf,
    )
    dtm = vec.fit_transform(texts)
    return dtm, vec


def evaluate_topics(topic_word_lists, texts, dictionary, label=""):
    """Calculate diversity, overlap, coherence."""
    n_topics = len(topic_word_lists)

    # Overlap
    word_sets = [set(words[:10]) for words in topic_word_lists]
    total_overlap = 0
    total_pairs = 0
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i+1, len(word_sets)):
            o = word_sets[i] & word_sets[j]
            total_overlap += len(o)
            total_pairs += 1
            if o:
                overlaps.append(f"  T{i+1}∩T{j+1}: {o}")
    avg_overlap = total_overlap / total_pairs if total_pairs > 0 else 0

    # Diversity
    all_words = []
    for ws in word_sets:
        all_words.extend(ws)
    unique = len(set(all_words))
    total = len(all_words)
    diversity = unique / total if total > 0 else 0

    # Coherence
    try:
        texts_tok = [doc.split() for doc in texts]
        cm = CoherenceModel(topics=topic_word_lists, texts=texts_tok, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
    except:
        coherence = 0

    return {
        'n_topics': n_topics,
        'diversity': diversity,
        'avg_overlap': avg_overlap,
        'coherence': coherence,
        'overlaps': overlaps,
    }


def run_lda(texts, stopwords, label, num_topics=10):
    """Run LDA with given topic count."""
    print(f"\n  --- LDA (k={num_topics}) on {label} ---")
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)

    lda = LDA(
        n_components=num_topics, random_state=42,
        doc_topic_prior=0.1/num_topics, topic_word_prior=0.01,
        learning_method='batch', max_iter=200,
    )
    lda.fit(dtm)

    # Extract topics
    topic_word_lists = []
    print(f"  Topics:")
    for i in range(num_topics):
        top_idx = lda.components_[i].argsort()[-15:][::-1]
        words = [features[j] for j in top_idx]
        probs = lda.components_[i][top_idx]
        probs = probs / probs.sum()
        topic_word_lists.append(words[:10])
        print(f"    T{i+1}: {', '.join(words[:10])}")

    metrics = evaluate_topics(topic_word_lists, texts, dictionary, label)

    if metrics['overlaps']:
        for o in metrics['overlaps'][:5]:
            print(f"    {o}")

    print(f"  Diversity: {metrics['diversity']:.3f} | Overlap: {metrics['avg_overlap']:.2f} | Coherence: {metrics['coherence']:.4f}")
    return metrics


def run_lda_auto(texts, stopwords, label):
    """Run LDA with automatic topic selection via coherence."""
    print(f"\n  --- LDA Auto-k on {label} ---")
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)
    texts_tok = [doc.split() for doc in texts]

    n = len(texts)
    k_range = range(5, min(26, max(10, n // 20)))

    best_k = 5
    best_coherence = -1
    results = []

    print(f"  Testing k={k_range.start}-{k_range.stop-1}...")
    for k in k_range:
        lda = LDA(
            n_components=k, random_state=42,
            doc_topic_prior=0.1/k, topic_word_prior=0.01,
            learning_method='batch', max_iter=100,
        )
        lda.fit(dtm)

        twl = []
        for i in range(k):
            top_idx = lda.components_[i].argsort()[-10:][::-1]
            twl.append([features[j] for j in top_idx])

        try:
            cm = CoherenceModel(topics=twl, texts=texts_tok, dictionary=dictionary, coherence='c_v')
            coh = cm.get_coherence()
        except:
            coh = 0

        results.append((k, coh))
        marker = " <<<" if coh > best_coherence else ""
        print(f"    k={k:2d}: coherence={coh:.4f}{marker}")

        if coh > best_coherence:
            best_coherence = coh
            best_k = k

    print(f"  Best k={best_k} (coherence={best_coherence:.4f})")

    # Re-run with best k and full iterations
    return run_lda(texts, stopwords, label, num_topics=best_k)


def run_hdp(texts, stopwords, label):
    """Run HDP with current best params."""
    print(f"\n  --- HDP on {label} ---")
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)

    n = len(texts)
    if n < 100: T, gamma = 20, 0.5
    elif n < 1000: T, gamma = 30, 1.0
    else: T, gamma = 50, 1.5

    hdp = HdpModel(
        corpus=corpus, id2word=dictionary, random_state=42,
        alpha=1, gamma=gamma, eta=0.01, T=T, K=8,
        kappa=0.7, tau=256.0, chunksize=min(256, n),
    )

    all_topics = hdp.get_topics()
    lda_alpha, _ = hdp.hdp_to_lda()
    alive = set(i for i, a in enumerate(lda_alpha) if a > 0.01)

    # Prevalence
    prevs = []
    for i in range(len(all_topics)):
        if i not in alive: continue
        ws = [dict(hdp[d]).get(i, 0) for d in corpus]
        prevs.append((i, np.mean(ws)))
    prevs.sort(key=lambda x: x[1], reverse=True)

    max_topics = max(5, min(25, int(5 + 5 * math.log10(max(n, 10)))))
    sig = [(i, p) for i, p in prevs[:max_topics] if p > 0.02]
    if len(sig) < 3 and len(prevs) >= 3:
        sig = prevs[:3]

    # Flat filter
    flat_thresh = max(0.003, 10.0 / len(features))
    filtered = []
    for idx, prev in sig:
        tw = hdp.show_topic(idx, topn=15)
        if tw and tw[0][1] >= flat_thresh:
            filtered.append((idx, prev))
    sig = filtered

    # Weak filter
    strong = []
    for idx, prev in sig:
        max_w = max((dict(hdp[d]).get(idx, 0) for d in corpus), default=0)
        if max_w >= 0.10:
            strong.append((idx, prev))
    sig = strong

    # Merge similar
    word_sets = {}
    for idx, _ in sig:
        word_sets[idx] = set(w for w, _ in hdp.show_topic(idx, topn=10))
    merged = set()
    for i, (idx_i, _) in enumerate(sig):
        if idx_i in merged: continue
        for idx_j, _ in sig[i+1:]:
            if idx_j in merged: continue
            inter = word_sets[idx_i] & word_sets[idx_j]
            union = word_sets[idx_i] | word_sets[idx_j]
            if len(inter) / len(union) > 0.4:
                merged.add(idx_j)
    sig = [(i, p) for i, p in sig if i not in merged]

    # Cross-topic dedup
    wc = Counter()
    for idx, _ in sig:
        for w, _ in hdp.show_topic(idx, topn=10):
            wc[w] += 1
    cross = {w for w, c in wc.items() if c >= 3}

    # Show topics
    topic_word_lists = []
    print(f"  Topics ({len(sig)}):")
    for rank, (idx, prev) in enumerate(sig):
        raw = hdp.show_topic(idx, topn=20)
        words = [w for w, _ in raw if w not in cross][:10]
        topic_word_lists.append(words)
        print(f"    T{rank+1} ({prev:.1%}): {', '.join(words)}")

    metrics = evaluate_topics(topic_word_lists, texts, dictionary, label)

    if metrics['overlaps']:
        for o in metrics['overlaps'][:5]:
            print(f"    {o}")

    print(f"  Diversity: {metrics['diversity']:.3f} | Overlap: {metrics['avg_overlap']:.2f} | Coherence: {metrics['coherence']:.4f}")
    return metrics


if __name__ == '__main__':
    zips = [
        ("/Users/mugdha/Desktop/research/100_sorted_pdf.zip", "100 papers"),
        ("/Users/mugdha/Desktop/research/1000_sorted_pdf.zip", "1000 papers"),
    ]
    if '--all' in sys.argv:
        zips.append(("/Users/mugdha/Desktop/research/10000_sorted_pdf.zip", "10000 papers"))

    all_results = []

    for zpath, label in zips:
        if not os.path.exists(zpath):
            print(f"SKIP: {zpath}")
            continue

        print(f"\n{'='*80}")
        print(f"  LOADING: {label}")
        print(f"{'='*80}")

        t0 = time.time()
        texts, stopwords = load_corpus(zpath)
        print(f"  {len(texts)} valid documents ({time.time()-t0:.0f}s)")

        hdp_r = run_hdp(texts, stopwords, label)
        lda_r = run_lda_auto(texts, stopwords, label)

        all_results.append((label, hdp_r, lda_r))

    print(f"\n\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Corpus':<15s} | {'Model':<10s} | {'Topics':>6s} | {'Div':>5s} | {'Overlap':>7s} | {'Coh':>6s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}")
    for label, hdp_r, lda_r in all_results:
        print(f"  {label:<15s} | {'HDP':<10s} | {hdp_r['n_topics']:6d} | {hdp_r['diversity']:.3f} | {hdp_r['avg_overlap']:7.2f} | {hdp_r['coherence']:.4f}")
        print(f"  {'':<15s} | {'LDA-auto':<10s} | {lda_r['n_topics']:6d} | {lda_r['diversity']:.3f} | {lda_r['avg_overlap']:7.2f} | {lda_r['coherence']:.4f}")

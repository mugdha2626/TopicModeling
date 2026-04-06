"""
Test LDA across 100, 1000, and optionally 10000 papers.
Auto-selects best k via coherence, then shows full results.
"""
import sys, os, zipfile, tempfile, time
import numpy as np
sys.path.insert(0, 'src')

from preprocessing import (
    preprocess_text, extract_text_from_pdf, clean_pdf_text,
    get_all_stopwords, prepare_gensim_corpus
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import CoherenceModel
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)


def load_corpus(zip_path):
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


def find_best_k(texts, stopwords, k_min=5, k_max=25):
    """Run LDA for each k and pick the one with best coherence."""
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)
    texts_tok = [doc.split() for doc in texts]

    n = len(texts)
    k_max = min(k_max, max(10, n // 20))

    print(f"\n  Finding best k ({k_min}-{k_max})...")
    best_k, best_coh = k_min, -1
    scores = []

    for k in range(k_min, k_max + 1):
        lda = LDA(
            n_components=k, random_state=42,
            doc_topic_prior=0.1 / k, topic_word_prior=0.01,
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

        scores.append((k, coh))
        marker = " <<<" if coh > best_coh else ""
        print(f"    k={k:2d}: coherence={coh:.4f}{marker}")
        if coh > best_coh:
            best_coh = coh
            best_k = k

    print(f"  Best: k={best_k} (coherence={best_coh:.4f})")
    return best_k


def run_lda(texts, stopwords, label, num_topics):
    """Run LDA with the given k and show full results."""
    print(f"\n  Running LDA with k={num_topics}...")
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)

    lda = LDA(
        n_components=num_topics, random_state=42,
        doc_topic_prior=0.1 / num_topics, topic_word_prior=0.01,
        learning_method='batch', max_iter=200,
    )
    lda.fit(dtm)
    doc_topic = lda.transform(dtm)

    converged = lda.n_iter_ < 200
    print(f"  Converged: {converged} (iters={lda.n_iter_})")
    print(f"  Perplexity: {lda.perplexity(dtm):.2f}")
    print(f"  Vocab: {len(features)}")

    # Extract topics
    topic_word_lists = []
    print(f"\n  TOPICS:")
    print(f"  {'-'*60}")
    for i in range(num_topics):
        top_idx = lda.components_[i].argsort()[-15:][::-1]
        words = [features[j] for j in top_idx]
        probs = lda.components_[i][top_idx]
        probs_norm = probs / probs.sum()
        topic_word_lists.append(words[:10])

        # How many docs dominated by this topic
        dom = (doc_topic[:, i] > 0.3).sum()
        avg_str = doc_topic[:, i].mean()

        print(f"  T{i+1} ({dom} docs, avg={avg_str:.3f}):")
        print(f"    {', '.join(words[:10])}")

    # Diversity
    word_sets = [set(w[:10]) for w in topic_word_lists]
    all_w = []
    for ws in word_sets:
        all_w.extend(ws)
    unique = len(set(all_w))
    total = len(all_w)
    diversity = unique / total if total > 0 else 0

    # Overlap
    total_overlap = 0
    total_pairs = 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            o = word_sets[i] & word_sets[j]
            total_overlap += len(o)
            total_pairs += 1
            if o:
                print(f"    T{i+1}∩T{j+1}: {o}")
    avg_overlap = total_overlap / total_pairs if total_pairs > 0 else 0

    # Coherence
    try:
        texts_tok = [doc.split() for doc in texts]
        cm = CoherenceModel(topics=topic_word_lists, texts=texts_tok, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        per_topic = cm.get_coherence_per_topic()
        print(f"\n  Per-topic coherence:")
        for i, c in enumerate(per_topic):
            print(f"    T{i+1}: {c:.4f}")
    except Exception as e:
        coherence = 0
        print(f"  Coherence failed: {e}")

    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {label} | LDA k={num_topics}")
    print(f"  Docs: {len(texts)} | Vocab: {len(features)}")
    print(f"  Diversity: {diversity:.3f} | Overlap: {avg_overlap:.2f} | Coherence: {coherence:.4f}")
    print(f"  {'='*60}")

    return {
        'label': label, 'k': num_topics, 'docs': len(texts),
        'vocab': len(features), 'diversity': diversity,
        'avg_overlap': avg_overlap, 'coherence': coherence,
    }


if __name__ == '__main__':
    zips = [
        ("/Users/mugdha/Desktop/research/100_sorted_pdf.zip", "100 papers"),
        ("/Users/mugdha/Desktop/research/1000_sorted_pdf.zip", "1000 papers"),
    ]
    if '--all' in sys.argv:
        zips.append(("/Users/mugdha/Desktop/research/10000_sorted_pdf.zip", "10000 papers"))

    results = []
    for zpath, label in zips:
        if not os.path.exists(zpath):
            print(f"SKIP: {zpath}")
            continue

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        t0 = time.time()
        texts, stopwords = load_corpus(zpath)
        print(f"  Loaded {len(texts)} docs ({time.time()-t0:.0f}s)")

        best_k = find_best_k(texts, stopwords)
        r = run_lda(texts, stopwords, label, best_k)
        results.append(r)

    print(f"\n\n{'='*80}")
    print(f"  LDA COMPARISON ACROSS CORPUS SIZES")
    print(f"{'='*80}")
    for r in results:
        print(f"  {r['label']:<15s} | k={r['k']:2d} | docs={r['docs']:5d} | vocab={r['vocab']:5d} | div={r['diversity']:.3f} | overlap={r['avg_overlap']:.2f} | coh={r['coherence']:.4f}")

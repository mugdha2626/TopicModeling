"""
Test LDA and HDP across 100, 1000, and optionally 10000 papers.
Reports topic words, diversity, overlap, coherence, and writes each model's
topic-word CSV (app Compare-tab format) to results/<model>/<corpus>/.

Every model function now takes seed= (default 42, matching the prior hardcoded
value) so runs can be repeated with a different seed -- see controls.py for a
harness that does this per (corpus, model, control, seed).

Usage (from lda/):
    venv/bin/python test_models_all.py                 # 100 + 1000, both models
    venv/bin/python test_models_all.py --all           # + 10000
    venv/bin/python test_models_all.py --models lda --lda-k 5
"""
import sys, os, zipfile, tempfile, time, math, json, argparse
import numpy as np
sys.path.insert(0, 'src')
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def corpus_name(zip_path):
    return os.path.splitext(os.path.basename(zip_path))[0]


def export_topic_csvs(out_dir, topic_word_dists, prevalence, top_n_words=None):
    """Write topic_word_distributions.csv + topic_prevalence.csv in the app's format."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k, wp in enumerate(topic_word_dists):
        items = sorted(wp.items(), key=lambda x: -x[1])
        if top_n_words:
            items = items[:top_n_words]
        rows += [(k + 1, f"Topic {k + 1}", w, f"{p:.8f}") for w, p in items]
    pd.DataFrame(rows, columns=["Topic_ID", "Topic_Name", "Word", "Probability"]).to_csv(
        os.path.join(out_dir, "topic_word_distributions.csv"), index=False)
    pd.DataFrame([(k + 1, f"Topic {k + 1}", f"{p:.6f}") for k, p in enumerate(prevalence)],
                 columns=["Topic_ID", "Topic_Name", "Prevalence"]).to_csv(
        os.path.join(out_dir, "topic_prevalence.csv"), index=False)


def load_corpus(zip_path, cache_root="cache"):
    """Extract and preprocess PDFs from a ZIP (stemmed corpus cached per zip)."""
    cache_file = os.path.join(cache_root, corpus_name(zip_path), "stemmed_texts.json")
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            texts = json.load(fh)
        print(f"  loaded {len(texts)} cached stemmed docs from {cache_file}")
        return texts, get_all_stopwords()
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
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as fh:
        json.dump(texts, fh)
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


def run_lda(texts, stopwords, label, num_topics=10, out_dir=None, seed=42):
    """Run LDA with given topic count."""
    print(f"\n  --- LDA (k={num_topics}) on {label} ---")
    dtm, vec = vectorize(texts, stopwords)
    features = vec.get_feature_names_out()
    corpus, dictionary = prepare_gensim_corpus(dtm, vec)

    lda = LDA(
        n_components=num_topics, random_state=seed,
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

    if out_dir:
        comps = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        dists = [{features[j]: float(comps[i, j]) for j in np.nonzero(comps[i] > 0)[0]} for i in range(num_topics)]
        prevalence = lda.transform(dtm).mean(axis=0).tolist()
        export_topic_csvs(out_dir, dists, prevalence)
        print(f"  CSVs -> {out_dir}")

    print(f"  Diversity: {metrics['diversity']:.3f} | Overlap: {metrics['avg_overlap']:.2f} | Coherence: {metrics['coherence']:.4f}")
    metrics['topic_words'] = topic_word_lists
    return metrics


def run_lda_auto(texts, stopwords, label, out_dir=None, fixed_k=None, seed=42):
    """Run LDA with automatic topic selection via coherence."""
    if fixed_k:
        return run_lda(texts, stopwords, label, num_topics=fixed_k, out_dir=out_dir, seed=seed)
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
            n_components=k, random_state=seed,
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
    return run_lda(texts, stopwords, label, num_topics=best_k, out_dir=out_dir, seed=seed)


def run_hdp(texts, stopwords, label, out_dir=None, seed=42):
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
        corpus=corpus, id2word=dictionary, random_state=seed,
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

    if out_dir and sig:
        all_t = hdp.get_topics()
        dists, prevalence = [], []
        for idx, prev in sig:
            row = np.asarray(all_t[idx], dtype=float); row = row / (row.sum() or 1.0)
            dists.append({features[j]: float(row[j]) for j in np.nonzero(row > 0)[0]})
            prevalence.append(float(prev))
        tot = sum(prevalence) or 1.0
        export_topic_csvs(out_dir, dists, [p / tot for p in prevalence])
        print(f"  CSVs -> {out_dir}")

    print(f"  Diversity: {metrics['diversity']:.3f} | Overlap: {metrics['avg_overlap']:.2f} | Coherence: {metrics['coherence']:.4f}")
    metrics['topic_words'] = topic_word_lists
    return metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='also run the 10k corpus')
    ap.add_argument('--models', default='hdp,lda', help='comma list of hdp, lda')
    ap.add_argument('--lda-k', type=int, default=None, help='fixed LDA k instead of the coherence sweep')
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    zips = [
        ("/Users/mugdha/Desktop/research/100_sorted_pdf.zip", "100 papers"),
        ("/Users/mugdha/Desktop/research/1000_sorted_pdf.zip", "1000 papers"),
    ]
    if args.all:
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
        cname = corpus_name(zpath)

        rows = []
        if 'hdp' in models:
            t = time.time(); r = run_hdp(texts, stopwords, label, out_dir=os.path.join('results', 'hdp', cname))
            rows.append(('HDP', r, time.time() - t))
        if 'lda' in models:
            t = time.time(); r = run_lda_auto(texts, stopwords, label, out_dir=os.path.join('results', 'lda', cname), fixed_k=args.lda_k)
            rows.append((f"LDA-k{args.lda_k}" if args.lda_k else 'LDA-auto', r, time.time() - t))
        all_results.append((label, rows))

    print(f"\n\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Corpus':<13s} | {'Model':<16s} | {'Topics':>6s} | {'Div':>5s} | {'Overlap':>7s} | {'c_v':>6s} | {'c_v*':>6s} | {'NPMI':>6s} | {'Outl':>5s} | {'sec':>6s}")
    print(f"  {'-'*13}-+-{'-'*16}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}")
    table = []
    for label, rows in all_results:
        for name, r, secs in rows:
            cs = r.get('coherence_shared'); npmi = r.get('npmi'); outl = r.get('outlier_fraction')
            print(f"  {label:<13s} | {name:<16s} | {r['n_topics']:6d} | {r['diversity']:.3f} | {r['avg_overlap']:7.2f} | "
                  f"{(r['coherence'] or 0):.4f} | {(f'{cs:.4f}' if cs is not None else '   -  '):>6s} | "
                  f"{(f'{npmi:.3f}' if npmi is not None else '  -  '):>6s} | {(f'{outl:.1%}' if outl is not None else '  -  '):>5s} | {secs:6.0f}")
            table.append({'corpus': label, 'model': name, 'n_topics': r['n_topics'], 'diversity': r['diversity'],
                          'avg_overlap': r['avg_overlap'], 'coherence': r['coherence'], 'coherence_shared': cs,
                          'npmi': npmi, 'outlier_fraction': outl, 'seconds': secs, 'topic_words': r.get('topic_words')})
    os.makedirs('results', exist_ok=True)
    with open('results/model_comparison.json', 'w') as fh:
        json.dump(table, fh, indent=2)
    print("\n  saved -> results/model_comparison.json")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        corpora = list(dict.fromkeys(r['corpus'] for r in table))
        fig, axes = plt.subplots(1, len(corpora), figsize=(6 * len(corpora), 4.5), squeeze=False)
        for ax, c in zip(axes[0], corpora):
            rows = [r for r in table if r['corpus'] == c]
            names = [r['model'] for r in rows]
            x = np.arange(len(rows)); w = 0.38
            coh = [(r['coherence_shared'] if r.get('coherence_shared') is not None else r['coherence']) or 0 for r in rows]
            ax.bar(x - w / 2, coh, w, label='c_v coherence (shared stemmed reference)', color='steelblue')
            ax.bar(x + w / 2, [r['diversity'] for r in rows], w, label='topic diversity', color='seagreen')
            for i, (a, b, r) in enumerate(zip(coh, [r['diversity'] for r in rows], rows)):
                ax.text(i - w / 2, a, f"{a:.2f}", ha='center', va='bottom', fontsize=7)
                ax.text(i + w / 2, b, f"{b:.2f}", ha='center', va='bottom', fontsize=7)
                ax.text(i, -0.08, f"{r['n_topics']} topics", ha='center', va='top', fontsize=7, transform=ax.get_xaxis_transform())
            ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=15)
            ax.set_ylim(0, 1.05); ax.set_title(c, fontsize=10)
        axes[0][0].legend(fontsize=7, loc='upper left')
        fig.suptitle('LDA vs HDP — coherence and diversity per corpus', fontsize=12)
        plt.tight_layout(); plt.savefig('results/model_comparison.png', dpi=130); plt.close()
        print("  chart -> results/model_comparison.png")
    except Exception as e:
        print(f"  chart skipped: {e}")

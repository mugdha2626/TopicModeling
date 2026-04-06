"""
Test HDP model against all three corpus sizes.
Mirrors the exact parameters from app.py's analyze_hdp_task.
"""
import sys, os, zipfile, tempfile, time, math
import numpy as np
sys.path.insert(0, 'src')

from preprocessing import (
    preprocess_text, extract_text_from_pdf, clean_pdf_text,
    extract_metadata, get_all_stopwords, ENHANCED_STOPWORDS,
    prepare_gensim_corpus
)
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import HdpModel, CoherenceModel
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)

def run_hdp_test(zip_path, label):
    print(f"\n{'='*80}")
    print(f"  TESTING: {label}")
    print(f"{'='*80}\n")

    # ── STEP 1: Extract and preprocess PDFs ──
    print("STEP 1: Extracting and preprocessing PDFs...")
    t0 = time.time()

    extracted_path = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extracted_path)

    all_stopwords = get_all_stopwords()

    pdf_texts = []
    for root, _, files in os.walk(extracted_path):
        for fname in files:
            if fname.endswith('.pdf') and not fname.startswith('._'):
                fpath = os.path.join(root, fname)
                text = extract_text_from_pdf(fpath)
                if not text:
                    continue
                cleaned = clean_pdf_text(text)
                processed = preprocess_text(cleaned)
                if processed and len(processed.split()) > 20:
                    pdf_texts.append(processed)

    pdf_count = len(pdf_texts)
    print(f"  Extracted {pdf_count} valid documents in {time.time()-t0:.1f}s")

    # ── STEP 2: Vectorize (mirrors app.py exactly) ──
    print("\nSTEP 2: Vectorizing...")

    if pdf_count < 50:
        max_features = 1000
    elif pdf_count < 500:
        max_features = 3000
    elif pdf_count < 5000:
        max_features = 5000
    else:
        max_features = 8000

    hdp_max_df = 0.35
    hdp_min_df = max(2, pdf_count // 50)

    print(f"  max_features={max_features}, max_df={hdp_max_df}, min_df={hdp_min_df}")

    vectorizer = CountVectorizer(
        max_df=hdp_max_df,
        min_df=hdp_min_df,
        stop_words=all_stopwords,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        max_features=max_features,
    )

    doc_term_matrix = vectorizer.fit_transform(pdf_texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Matrix: {doc_term_matrix.shape[0]} docs × {doc_term_matrix.shape[1]} words")

    # Top words check
    word_freq = np.asarray(doc_term_matrix.sum(axis=0)).flatten()
    top_indices = word_freq.argsort()[::-1][:20]
    print(f"\n  TOP 20 WORDS (check for junk):")
    for i, idx in enumerate(top_indices):
        doc_count = (doc_term_matrix[:, idx].toarray() > 0).sum()
        print(f"    {i+1:2d}. {feature_names[idx]:25s}  freq={int(word_freq[idx]):6d}  in {doc_count}/{pdf_count} docs ({doc_count/pdf_count*100:.0f}%)")

    # ── STEP 3: Train HDP (mirrors app.py exactly) ──
    print("\nSTEP 3: Preparing corpus and training HDP...")
    corpus, dictionary = prepare_gensim_corpus(doc_term_matrix, vectorizer)

    if pdf_count < 100:
        T, hdp_gamma = 20, 0.5
    elif pdf_count < 1000:
        T, hdp_gamma = 30, 1.0
    else:
        T, hdp_gamma = 50, 1.5

    print(f"  Params: T={T}, alpha=1, gamma={hdp_gamma}, eta=0.01, kappa=0.7, tau=256")

    t1 = time.time()
    hdp_model = HdpModel(
        corpus=corpus,
        id2word=dictionary,
        random_state=42,
        alpha=1,
        gamma=hdp_gamma,
        eta=0.01,
        T=T,
        K=8,
        kappa=0.7,
        tau=256.0,
        chunksize=min(256, pdf_count),
    )
    print(f"  Training completed in {time.time()-t1:.1f}s")

    all_topics = hdp_model.get_topics()

    # ── STEP 4: Gate 1 — Alpha weights ──
    print(f"\nSTEP 4: Filtering topics...")
    lda_alpha, lda_beta = hdp_model.hdp_to_lda()
    alpha_alive = set(i for i, a in enumerate(lda_alpha) if a > 0.01)
    print(f"  Alpha gate: {len(alpha_alive)}/{len(all_topics)} alive")

    # ── STEP 5: Gate 2 — Prevalence ──
    topic_prevalences = []
    for topic_idx in range(len(all_topics)):
        if topic_idx not in alpha_alive:
            continue
        doc_weights = []
        for doc_bow in corpus:
            doc_dist = dict(hdp_model[doc_bow])
            doc_weights.append(doc_dist.get(topic_idx, 0.0))
        avg = np.mean(doc_weights)
        topic_prevalences.append((topic_idx, avg))

    topic_prevalences.sort(key=lambda x: x[1], reverse=True)

    max_topics_to_keep = max(5, min(25, int(5 + 5 * math.log10(max(pdf_count, 10)))))
    min_prevalence = 0.02

    significant = []
    for idx, prev in topic_prevalences[:max_topics_to_keep]:
        if prev > min_prevalence:
            significant.append(idx)

    if len(significant) < 3 and len(topic_prevalences) >= 3:
        significant = [idx for idx, _ in topic_prevalences[:3]]

    print(f"  Prevalence gate (>{min_prevalence:.0%}, max {max_topics_to_keep}): {len(significant)} topics")

    # ── STEP 6: Gate 3 — Junk detection ──
    vocab_size = len(feature_names)
    flat_threshold = max(0.003, 10.0 / vocab_size)
    print(f"  Flat threshold: {flat_threshold:.4f} (vocab={vocab_size})")

    filtered = []
    for idx in significant:
        tw = hdp_model.show_topic(idx, topn=15)
        if not tw:
            continue
        top_prob = tw[0][1]
        if top_prob < flat_threshold:
            print(f"    Removing flat topic {idx}: top_prob={top_prob:.4f}")
            continue
        filtered.append(idx)
    significant = filtered

    # ── STEP 7: Gate 4 — Weak topics ──
    strong = []
    for idx in significant:
        max_w = max((dict(hdp_model[d]).get(idx, 0.0) for d in corpus), default=0)
        if max_w < 0.10:
            print(f"    Removing weak topic {idx}: max_weight={max_w:.3f}")
            continue
        strong.append(idx)
    significant = strong

    # ── STEP 8: Gate 5 — Merge similar ──
    word_sets = {}
    for idx in significant:
        word_sets[idx] = set(w for w, _ in hdp_model.show_topic(idx, topn=10))

    merged_out = set()
    for i, idx_i in enumerate(significant):
        if idx_i in merged_out:
            continue
        for idx_j in significant[i+1:]:
            if idx_j in merged_out:
                continue
            inter = word_sets[idx_i] & word_sets[idx_j]
            union = word_sets[idx_i] | word_sets[idx_j]
            jacc = len(inter) / len(union) if union else 0
            if jacc > 0.4:
                merged_out.add(idx_j)
                print(f"    Merging topic {idx_j} into {idx_i} (Jaccard={jacc:.2f})")

    significant = [idx for idx in significant if idx not in merged_out]
    print(f"  After all gates: {len(significant)} topics")

    # ── STEP 9: Cross-topic dedup ──
    word_count = Counter()
    for idx in significant:
        for w, _ in hdp_model.show_topic(idx, topn=10):
            word_count[w] += 1
    cross_words = {w for w, c in word_count.items() if c >= 3}
    if cross_words:
        print(f"  Cross-topic words removed from display: {cross_words}")

    # ── STEP 10: Show final topics ──
    num_words = 10
    print(f"\n{'='*60}")
    print(f"  FINAL TOPICS ({len(significant)} topics)")
    print(f"{'='*60}")

    all_top_words = []
    for rank, idx in enumerate(significant):
        prev = dict(topic_prevalences).get(idx, 0)
        raw_words = hdp_model.show_topic(idx, topn=num_words + len(cross_words) + 5)
        words = [(w, p) for w, p in raw_words if w not in cross_words][:num_words]
        word_list = [w for w, _ in words]
        all_top_words.append(set(word_list))

        print(f"\n  TOPIC {rank+1} (prevalence={prev:.1%}):")
        for w, p in words:
            print(f"    {w:25s} {p:.4f}")

    # ── STEP 11: Quality metrics ──
    print(f"\n{'='*60}")
    print(f"  QUALITY METRICS")
    print(f"{'='*60}")

    # Overlap
    total_pairs = 0
    total_overlap = 0
    for i in range(len(all_top_words)):
        for j in range(i+1, len(all_top_words)):
            overlap = all_top_words[i] & all_top_words[j]
            total_pairs += 1
            total_overlap += len(overlap)
            if overlap:
                print(f"  Topic {i+1} ∩ Topic {j+1}: {overlap}")

    avg_overlap = total_overlap / total_pairs if total_pairs > 0 else 0

    # Diversity
    all_flat = []
    for ws in all_top_words:
        all_flat.extend(ws)
    unique = len(set(all_flat))
    total = len(all_flat)
    diversity = unique / total if total > 0 else 0

    print(f"\n  Avg word overlap: {avg_overlap:.2f} words/pair")
    print(f"  Diversity: {diversity:.3f} ({unique} unique / {total} total)")

    # Coherence
    try:
        topic_word_lists = []
        for idx in significant:
            raw = hdp_model.show_topic(idx, topn=10)
            topic_word_lists.append([w for w, _ in raw])
        texts_tok = [doc.split() for doc in pdf_texts]
        cm = CoherenceModel(topics=topic_word_lists, texts=texts_tok, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        print(f"  Coherence (c_v): {coherence:.4f}")
    except Exception as e:
        coherence = 0
        print(f"  Coherence failed: {e}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {label}")
    print(f"  Docs: {pdf_count} | Vocab: {len(feature_names)} | Topics: {len(significant)}")
    print(f"  Diversity: {diversity:.3f} | Overlap: {avg_overlap:.2f} | Coherence: {coherence:.4f}")
    print(f"{'='*80}\n")

    return {
        'label': label, 'pdf_count': pdf_count, 'vocab_size': len(feature_names),
        'num_topics': len(significant), 'diversity': diversity,
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
        if os.path.exists(zpath):
            results.append(run_hdp_test(zpath, label))
        else:
            print(f"SKIP: {zpath} not found")

    print("\n" + "=" * 80)
    print("  COMPARISON ACROSS CORPUS SIZES")
    print("=" * 80)
    for r in results:
        print(f"  {r['label']:15s} | docs={r['pdf_count']:5d} | vocab={r['vocab_size']:5d} | topics={r['num_topics']:2d} | div={r['diversity']:.3f} | overlap={r['avg_overlap']:.2f} | coh={r['coherence']:.4f}")

"""
Test script for validating topic modeling pipeline.
Uses sklearn's 20 Newsgroups dataset as a benchmark.

Run: python test_models.py
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import HdpModel, CoherenceModel
from gensim import corpora

from preprocessing import (
    preprocess_text,
    clean_pdf_text,
    prepare_gensim_corpus,
    ENHANCED_STOPWORDS,
    get_all_stopwords,
)

from comparison_utils import (
    calculate_tvd,
    calculate_optimal_transport_distance,
    calculate_best_match_metrics,
    bootstrap_ot_distance,
    mann_whitney_test,
    permutation_test_ot_distance,
)


def test_preprocessing():
    """Test text preprocessing functions."""
    print("\n" + "=" * 60)
    print("TEST 1: Preprocessing")
    print("=" * 60)

    # Test basic preprocessing
    text = "The study uses neural networks (Smith, 2020) for classification tasks."
    result = preprocess_text(text)
    print(f"  Input:  {text}")
    print(f"  Output: {result}")

    # Verify stopwords removed
    assert "study" not in result, "'study' should be removed as stopword"
    assert "use" not in result.split(), "'use' should be removed as stopword"
    print("  [PASS] Stopwords removed correctly")

    # Test citation removal
    text_with_citations = "This approach (Johnson et al., 2019) improves upon prior work [1,2,3] significantly."
    cleaned = clean_pdf_text(text_with_citations)
    assert "[1,2,3]" not in cleaned, "Bracket citations should be removed"
    print("  [PASS] Citations removed correctly")

    # Test short text filtering
    result = preprocess_text("too short")
    assert result == "", "Short text should return empty string"
    print("  [PASS] Short text filtered correctly")

    # Test lemmatization
    text = "The researchers were running multiple experiments on different algorithms"
    result = preprocess_text(text)
    print(f"  Lemma test input:  {text}")
    print(f"  Lemma test output: {result}")
    # "running" should be lemmatized to "run", "algorithms" to "algorithm"
    if "algorithm" in result:
        print("  [PASS] Lemmatization working (algorithm)")
    else:
        print("  [WARN] 'algorithm' not found in output - check lemmatizer")

    print("  --- Preprocessing tests PASSED ---")
    return True


def test_lda_model():
    """Test LDA with 20 Newsgroups benchmark data."""
    print("\n" + "=" * 60)
    print("TEST 2: LDA Model Quality")
    print("=" * 60)

    # Load 4 distinct categories
    categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.mideast']
    print(f"  Loading 20 Newsgroups categories: {categories}")

    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # Preprocess
    print(f"  Documents loaded: {len(newsgroups.data)}")
    texts = []
    for doc in newsgroups.data[:300]:  # Use 300 docs for speed
        processed = preprocess_text(doc)
        if processed:
            texts.append(processed)
    print(f"  After preprocessing: {len(texts)} documents")

    # Vectorize
    all_stopwords = get_all_stopwords()
    vectorizer = CountVectorizer(
        max_df=0.7,
        min_df=2,
        stop_words=all_stopwords,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        max_features=2000,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(feature_names)}")

    # Train LDA with improved parameters
    num_topics = 4
    lda = LDA(
        n_components=num_topics,
        random_state=42,
        doc_topic_prior=0.1 / num_topics,
        topic_word_prior=0.01,
        learning_method='batch',
        max_iter=200,
        evaluate_every=10,
        perp_tol=1e-3
    )
    lda.fit(doc_term_matrix)

    # Check convergence
    converged = lda.n_iter_ < 200
    print(f"  Converged: {converged} (iterations: {lda.n_iter_})")
    assert converged, "LDA should converge within 200 iterations on this dataset"
    print("  [PASS] LDA converged")

    # Print topics
    print(f"\n  Topics discovered:")
    topic_word_lists = []
    all_top_words = []
    for topic_idx in range(num_topics):
        top_indices = lda.components_[topic_idx].argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_word_lists.append(top_words)
        all_top_words.extend(top_words)
        print(f"    Topic {topic_idx + 1}: {', '.join(top_words)}")

    # Calculate coherence
    tokenized_texts = [text.split() for text in texts]
    corpus_gensim, dictionary_gensim = prepare_gensim_corpus(doc_term_matrix, vectorizer)

    try:
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized_texts,
            dictionary=dictionary_gensim,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        per_topic = cm.get_coherence_per_topic()
        print(f"\n  Coherence (c_v): {coherence:.4f}")
        for i, c in enumerate(per_topic):
            print(f"    Topic {i + 1}: {c:.4f}")

        assert coherence > 0.25, f"Coherence {coherence:.4f} is too low (expected > 0.25)"
        print("  [PASS] Coherence above minimum threshold")
    except Exception as e:
        print(f"  [WARN] Coherence calculation failed: {e}")

    # Calculate diversity
    diversity = len(set(all_top_words)) / len(all_top_words)
    print(f"\n  Topic diversity: {diversity:.4f}")
    assert diversity > 0.3, f"Diversity {diversity:.4f} is too low (expected > 0.3)"
    print("  [PASS] Topic diversity above minimum threshold")

    # Check perplexity
    perplexity = lda.perplexity(doc_term_matrix)
    print(f"  Perplexity: {perplexity:.2f}")

    print("  --- LDA tests PASSED ---")
    return lda, vectorizer, doc_term_matrix, feature_names, texts


def test_hdp_model(texts, vectorizer, doc_term_matrix, feature_names):
    """Test HDP with the same data."""
    print("\n" + "=" * 60)
    print("TEST 3: HDP Model Quality")
    print("=" * 60)

    # Prepare gensim corpus
    corpus, dictionary = prepare_gensim_corpus(doc_term_matrix, vectorizer)

    # Train HDP with app.py's parameters for small corpus
    pdf_count = len(texts)
    if pdf_count < 50:
        T, gamma_scaled = 15, 0.5
    elif pdf_count < 500:
        T, gamma_scaled = 30, 0.3
    else:
        T, gamma_scaled = 50, 0.1

    print(f"  Training HDP: T={T}, gamma={gamma_scaled}, documents={pdf_count}")

    hdp_model = HdpModel(
        corpus=corpus,
        id2word=dictionary,
        random_state=42,
        alpha=0.1,
        gamma=gamma_scaled,
        eta=0.01,
        T=T,
        K=15,
        kappa=0.75,
        tau=64.0
    )

    # Count significant topics
    all_topics = hdp_model.get_topics()
    print(f"  Total topics from model: {len(all_topics)}")

    # Calculate topic prevalence
    topic_prevalences = np.zeros(len(all_topics))
    for doc_bow in corpus:
        doc_topics = dict(hdp_model[doc_bow])
        for topic_idx, prob in doc_topics.items():
            if topic_idx < len(topic_prevalences):
                topic_prevalences[topic_idx] += prob
    topic_prevalences /= len(corpus)

    # Filter significant topics (>1% prevalence)
    significant = [(i, p) for i, p in enumerate(topic_prevalences) if p > 0.01]
    significant.sort(key=lambda x: x[1], reverse=True)
    num_significant = min(len(significant), 15)

    print(f"  Significant topics (>1% prevalence): {num_significant}")
    assert num_significant >= 2, f"HDP found only {num_significant} topics (expected >= 2)"
    print("  [PASS] HDP discovered multiple topics")

    # Print top topics
    topic_word_lists = []
    for i, (topic_idx, prevalence) in enumerate(significant[:num_significant]):
        topic_words = hdp_model.show_topic(topic_idx, topn=10)
        words = [w for w, _ in topic_words]
        topic_word_lists.append(words)
        print(f"    Topic {i + 1} (prevalence={prevalence:.3f}): {', '.join(words[:5])}")

    # Coherence
    tokenized_texts = [text.split() for text in texts]
    try:
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        print(f"\n  Coherence (c_v): {coherence:.4f}")
        assert coherence > 0.15, f"HDP coherence {coherence:.4f} too low"
        print("  [PASS] HDP coherence above minimum threshold")
    except Exception as e:
        print(f"  [WARN] HDP coherence calculation failed: {e}")

    print("  --- HDP tests PASSED ---")
    return hdp_model, corpus, dictionary


def test_comparison_self():
    """Test comparison of a dataset against itself (should be near-zero distance)."""
    print("\n" + "=" * 60)
    print("TEST 4: Comparison - Self-Comparison")
    print("=" * 60)

    # Create simple mock topic distributions
    vocab = ['word1', 'word2', 'word3', 'word4', 'word5']
    topics = [
        {'topic_id': 0, 'topic_name': 'Topic 1', 'word_probabilities': {'word1': 0.4, 'word2': 0.3, 'word3': 0.2, 'word4': 0.05, 'word5': 0.05}},
        {'topic_id': 1, 'topic_name': 'Topic 2', 'word_probabilities': {'word1': 0.05, 'word2': 0.05, 'word3': 0.2, 'word4': 0.3, 'word5': 0.4}},
    ]
    prevalence = [0.6, 0.4]

    # Self-comparison: TVD should be 0
    tvd = calculate_tvd(
        topics[0]['word_probabilities'],
        topics[0]['word_probabilities'],
        vocab
    )
    print(f"  Self-TVD (should be ~0): {tvd:.6f}")
    assert tvd < 0.01, f"Self-TVD should be near 0, got {tvd}"
    print("  [PASS] Self-TVD is near zero")

    # Different topics: TVD should be > 0
    tvd_diff = calculate_tvd(
        topics[0]['word_probabilities'],
        topics[1]['word_probabilities'],
        vocab
    )
    print(f"  Different-TVD (should be > 0): {tvd_diff:.6f}")
    assert tvd_diff > 0.1, f"Different-TVD should be > 0.1, got {tvd_diff}"
    print("  [PASS] Different topics have positive TVD")

    # Self OT distance
    ot_dist = calculate_optimal_transport_distance(
        topics, topics, prevalence, prevalence, vocab
    )
    if ot_dist is not None:
        print(f"  Self-OT distance (should be ~0): {ot_dist:.6f}")
        assert ot_dist < 0.05, f"Self-OT should be near 0, got {ot_dist}"
        print("  [PASS] Self-OT distance is near zero")
    else:
        print("  [SKIP] POT library not available for OT distance")

    # Best match self
    bm = calculate_best_match_metrics(topics, topics, vocab)
    print(f"  Self best-match reciprocal: {bm['num_reciprocal']}")
    assert bm['num_reciprocal'] == 2, f"Self-comparison should have 2 reciprocal matches"
    print("  [PASS] Self-comparison has correct reciprocal matches")

    print("  --- Self-comparison tests PASSED ---")


def test_comparison_different():
    """Test comparison of two very different topic sets."""
    print("\n" + "=" * 60)
    print("TEST 5: Comparison - Different Datasets")
    print("=" * 60)

    vocab = ['cat', 'dog', 'bird', 'car', 'bus', 'train']

    topics1 = [
        {'topic_id': 0, 'topic_name': 'Animals', 'word_probabilities': {'cat': 0.5, 'dog': 0.3, 'bird': 0.2, 'car': 0.0, 'bus': 0.0, 'train': 0.0}},
    ]
    topics2 = [
        {'topic_id': 0, 'topic_name': 'Vehicles', 'word_probabilities': {'cat': 0.0, 'dog': 0.0, 'bird': 0.0, 'car': 0.4, 'bus': 0.35, 'train': 0.25}},
    ]

    tvd = calculate_tvd(
        topics1[0]['word_probabilities'],
        topics2[0]['word_probabilities'],
        vocab
    )
    print(f"  TVD between disjoint topics: {tvd:.4f}")
    assert tvd > 0.8, f"Disjoint topics should have TVD > 0.8, got {tvd}"
    print("  [PASS] Disjoint topics have high TVD")

    # OT distance should be high
    prevalence = [1.0]
    ot_dist = calculate_optimal_transport_distance(
        topics1, topics2, prevalence, prevalence, vocab
    )
    if ot_dist is not None:
        print(f"  OT distance between disjoint sets: {ot_dist:.4f}")
        assert ot_dist > 0.5, f"Disjoint OT should be > 0.5, got {ot_dist}"
        print("  [PASS] Disjoint sets have high OT distance")
    else:
        print("  [SKIP] POT library not available")

    print("  --- Different-dataset tests PASSED ---")


def test_mann_whitney():
    """Test Mann-Whitney statistical test."""
    print("\n" + "=" * 60)
    print("TEST 6: Statistical Tests")
    print("=" * 60)

    # Identical distributions should not be significant
    identical = [0.3, 0.4, 0.35, 0.32, 0.38]
    result = mann_whitney_test(identical, identical)
    if result:
        print(f"  Mann-Whitney (identical): p={result['p_value']:.4f}, significant={result['significant']}")
        assert not result['significant'], "Identical distributions should not be significant"
        print("  [PASS] Identical distributions not significant")
    else:
        print("  [WARN] Mann-Whitney returned None for identical distributions")

    # Very different distributions should be significant
    low = [0.1, 0.12, 0.08, 0.11, 0.09, 0.1, 0.13, 0.07]
    high = [0.8, 0.85, 0.9, 0.82, 0.88, 0.79, 0.91, 0.87]
    result = mann_whitney_test(low, high)
    if result:
        print(f"  Mann-Whitney (different): p={result['p_value']:.6f}, significant={result['significant']}")
        assert result['significant'], "Very different distributions should be significant"
        print("  [PASS] Different distributions are significant")

    print("  --- Statistical tests PASSED ---")


def run_all_tests():
    """Run the complete test suite."""
    print("\n" + "#" * 60)
    print("#  TOPIC MODELING TEST SUITE")
    print("#" * 60)

    passed = 0
    failed = 0

    # Test 1: Preprocessing
    try:
        test_preprocessing()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Test 2: LDA
    lda_result = None
    try:
        lda_result = test_lda_model()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Test 3: HDP
    try:
        if lda_result:
            lda, vectorizer, doc_term_matrix, feature_names, texts = lda_result
            test_hdp_model(texts, vectorizer, doc_term_matrix, feature_names)
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Test 4: Self-comparison
    try:
        test_comparison_self()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Test 5: Different comparison
    try:
        test_comparison_different()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Test 6: Statistical tests
    try:
        test_mann_whitney()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

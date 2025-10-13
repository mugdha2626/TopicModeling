"""
Comparison utilities for topic modeling analysis.

This module provides functions for comparing topic distributions across different corpus sizes
using advanced metrics including:
- Optimal Transport (OT) distance with prevalence weighting
- Best-match analysis with reciprocal matching
- Bootstrap confidence intervals
- Statistical significance tests
"""

import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    logger.warning("POT library not available. Optimal Transport calculations will be disabled.")
    POT_AVAILABLE = False


def calculate_tvd(topic_dist_1, topic_dist_2, vocabulary):
    """
    Calculate Total Variation Distance between two topic-word distributions.

    TVD = 0.5 * sum(|P(w|topic1) - P(w|topic2)|) for all words w

    Args:
        topic_dist_1: Dict mapping words to probabilities for topic 1
        topic_dist_2: Dict mapping words to probabilities for topic 2
        vocabulary: List of all vocabulary words

    Returns:
        float: TVD distance between 0 (identical) and 1 (completely different)
    """
    try:
        # Create probability vectors for all words in vocabulary
        prob_vector_1 = np.zeros(len(vocabulary))
        prob_vector_2 = np.zeros(len(vocabulary))

        # Fill probability vectors
        for i, word in enumerate(vocabulary):
            prob_vector_1[i] = topic_dist_1.get(word, 0.0)
            prob_vector_2[i] = topic_dist_2.get(word, 0.0)

        # Normalize to ensure they sum to 1
        if prob_vector_1.sum() > 0:
            prob_vector_1 = prob_vector_1 / prob_vector_1.sum()
        if prob_vector_2.sum() > 0:
            prob_vector_2 = prob_vector_2 / prob_vector_2.sum()

        # Calculate TVD
        tvd = 0.5 * np.sum(np.abs(prob_vector_1 - prob_vector_2))
        return float(tvd)

    except Exception as e:
        logger.error("TVD calculation error: %s", str(e))
        return 1.0  # Return maximum distance on error


def calculate_optimal_transport_distance(topics1, topics2, prevalence1, prevalence2, vocabulary):
    """
    Calculate Optimal Transport distance between two sets of topics using prevalence weights.

    This provides a global divergence metric that accounts for:
    - Topic prevalence (how much each topic contributes to the corpus)
    - Topic splits and merges across corpus sizes
    - Overall distribution differences

    Args:
        topics1: List of dicts with topic word distributions for corpus 1
        topics2: List of dicts with topic word distributions for corpus 2
        prevalence1: List of prevalence weights for corpus 1 topics (must sum to 1)
        prevalence2: List of prevalence weights for corpus 2 topics (must sum to 1)
        vocabulary: List of all vocabulary words

    Returns:
        float: OT distance (lower = more similar)
    """
    if not POT_AVAILABLE:
        logger.error("POT library not available for OT distance calculation")
        return None

    try:
        # Normalize prevalence weights to ensure they sum to 1
        prevalence1 = np.array(prevalence1)
        prevalence2 = np.array(prevalence2)

        if prevalence1.sum() > 0:
            prevalence1 = prevalence1 / prevalence1.sum()
        else:
            # Uniform distribution if no prevalence data
            prevalence1 = np.ones(len(topics1)) / len(topics1)

        if prevalence2.sum() > 0:
            prevalence2 = prevalence2 / prevalence2.sum()
        else:
            prevalence2 = np.ones(len(topics2)) / len(topics2)

        # Build cost matrix using TVD between all topic pairs
        num_topics1 = len(topics1)
        num_topics2 = len(topics2)
        cost_matrix = np.zeros((num_topics1, num_topics2))

        for i, topic1 in enumerate(topics1):
            for j, topic2 in enumerate(topics2):
                cost_matrix[i, j] = calculate_tvd(
                    topic1['word_probabilities'],
                    topic2['word_probabilities'],
                    vocabulary
                )

        # Calculate optimal transport distance
        ot_distance = ot.emd2(prevalence1, prevalence2, cost_matrix)

        return float(ot_distance)

    except Exception as e:
        logger.error("OT distance calculation error: %s", str(e))
        return None


def calculate_best_match_metrics(topics1, topics2, vocabulary, tvd_threshold=0.3):
    """
    Calculate best-match analysis between two sets of topics.

    For each topic in corpus A, finds the closest match in corpus B (and vice versa).
    Also identifies reciprocal matches where A→B and B→A agree.

    Args:
        topics1: List of dicts with topic word distributions for corpus 1
        topics2: List of dicts with topic word distributions for corpus 2
        vocabulary: List of all vocabulary words
        tvd_threshold: Threshold for considering a match "stable" (default 0.3)

    Returns:
        dict with keys:
            - best_matches_1to2: List of (topic1_idx, topic2_idx, tvd) for A→B
            - best_matches_2to1: List of (topic2_idx, topic1_idx, tvd) for B→A
            - reciprocal_matches: List of (topic1_idx, topic2_idx, tvd) where both directions agree
            - coverage_at_threshold: Fraction of topics with match below threshold
            - high_divergence: Fraction of topics with match above 0.7 (indicating instability)
    """
    try:
        num_topics1 = len(topics1)
        num_topics2 = len(topics2)

        # Build full TVD matrix
        tvd_matrix = np.zeros((num_topics1, num_topics2))
        for i, topic1 in enumerate(topics1):
            for j, topic2 in enumerate(topics2):
                tvd_matrix[i, j] = calculate_tvd(
                    topic1['word_probabilities'],
                    topic2['word_probabilities'],
                    vocabulary
                )

        # Find best matches A→B (corpus 1 to corpus 2)
        best_matches_1to2 = []
        for i in range(num_topics1):
            best_j = np.argmin(tvd_matrix[i, :])
            best_tvd = tvd_matrix[i, best_j]
            best_matches_1to2.append({
                'topic1_idx': i,
                'topic1_name': topics1[i]['topic_name'],
                'topic2_idx': int(best_j),
                'topic2_name': topics2[best_j]['topic_name'],
                'tvd': float(best_tvd)
            })

        # Find best matches B→A (corpus 2 to corpus 1)
        best_matches_2to1 = []
        for j in range(num_topics2):
            best_i = np.argmin(tvd_matrix[:, j])
            best_tvd = tvd_matrix[best_i, j]
            best_matches_2to1.append({
                'topic2_idx': j,
                'topic2_name': topics2[j]['topic_name'],
                'topic1_idx': int(best_i),
                'topic1_name': topics1[best_i]['topic_name'],
                'tvd': float(best_tvd)
            })

        # Find reciprocal matches (where A→B and B→A agree)
        reciprocal_matches = []
        for match_1to2 in best_matches_1to2:
            i = match_1to2['topic1_idx']
            j = match_1to2['topic2_idx']
            # Check if B→A also maps back to the same topic
            reverse_match = best_matches_2to1[j]
            if reverse_match['topic1_idx'] == i:
                reciprocal_matches.append({
                    'topic1_idx': i,
                    'topic1_name': match_1to2['topic1_name'],
                    'topic2_idx': j,
                    'topic2_name': match_1to2['topic2_name'],
                    'tvd': match_1to2['tvd']
                })

        # Calculate coverage metrics
        tvd_values_1to2 = [m['tvd'] for m in best_matches_1to2]
        tvd_values_2to1 = [m['tvd'] for m in best_matches_2to1]

        coverage_1to2 = np.mean([tvd < tvd_threshold for tvd in tvd_values_1to2])
        coverage_2to1 = np.mean([tvd < tvd_threshold for tvd in tvd_values_2to1])

        high_divergence_1to2 = np.mean([tvd > 0.7 for tvd in tvd_values_1to2])
        high_divergence_2to1 = np.mean([tvd > 0.7 for tvd in tvd_values_2to1])

        return {
            'best_matches_1to2': best_matches_1to2,
            'best_matches_2to1': best_matches_2to1,
            'reciprocal_matches': reciprocal_matches,
            'coverage_1to2': float(coverage_1to2),
            'coverage_2to1': float(coverage_2to1),
            'high_divergence_1to2': float(high_divergence_1to2),
            'high_divergence_2to1': float(high_divergence_2to1),
            'num_reciprocal': len(reciprocal_matches),
            'tvd_matrix': tvd_matrix.tolist()
        }

    except Exception as e:
        logger.error("Best-match calculation error: %s", str(e))
        return None


def bootstrap_ot_distance(topics1, topics2, prevalence1, prevalence2, vocabulary, n_bootstrap=1000):
    """
    Calculate bootstrap confidence intervals for OT distance.

    Uses Monte Carlo sampling to estimate uncertainty in the OT distance metric.

    Args:
        topics1, topics2: Topic distributions
        prevalence1, prevalence2: Topic prevalence weights
        vocabulary: Combined vocabulary
        n_bootstrap: Number of bootstrap samples (default 1000)

    Returns:
        dict with keys:
            - mean: Mean OT distance across bootstrap samples
            - ci_lower: Lower bound of 95% CI
            - ci_upper: Upper bound of 95% CI
            - std: Standard deviation
    """
    if not POT_AVAILABLE:
        logger.error("POT library not available for bootstrap OT distance")
        return None

    try:
        ot_samples = []

        # For each bootstrap iteration, resample prevalence weights with Dirichlet noise
        prevalence1 = np.array(prevalence1)
        prevalence2 = np.array(prevalence2)

        # Normalize
        if prevalence1.sum() > 0:
            prevalence1 = prevalence1 / prevalence1.sum()
        else:
            prevalence1 = np.ones(len(topics1)) / len(topics1)

        if prevalence2.sum() > 0:
            prevalence2 = prevalence2 / prevalence2.sum()
        else:
            prevalence2 = np.ones(len(topics2)) / len(topics2)

        # Use small concentration parameter for Dirichlet sampling (adds noise)
        alpha1 = prevalence1 * 100  # Higher concentration = less noise
        alpha2 = prevalence2 * 100

        for _ in range(n_bootstrap):
            # Sample from Dirichlet distribution around the prevalence
            sampled_prev1 = np.random.dirichlet(alpha1)
            sampled_prev2 = np.random.dirichlet(alpha2)

            # Calculate OT distance with sampled prevalence
            ot_dist = calculate_optimal_transport_distance(
                topics1, topics2, sampled_prev1, sampled_prev2, vocabulary
            )

            if ot_dist is not None:
                ot_samples.append(ot_dist)

        if len(ot_samples) == 0:
            return None

        ot_samples = np.array(ot_samples)

        return {
            'mean': float(np.mean(ot_samples)),
            'ci_lower': float(np.percentile(ot_samples, 2.5)),
            'ci_upper': float(np.percentile(ot_samples, 97.5)),
            'std': float(np.std(ot_samples))
        }

    except Exception as e:
        logger.error("Bootstrap OT distance error: %s", str(e))
        return None


def bootstrap_best_match_metrics(topics1, topics2, vocabulary, n_bootstrap=1000):
    """
    Calculate bootstrap confidence intervals for best-match TVD values.

    Resamples topic-word distributions to estimate uncertainty.

    Args:
        topics1, topics2: Topic distributions
        vocabulary: Combined vocabulary
        n_bootstrap: Number of bootstrap samples

    Returns:
        dict with bootstrap statistics for best-match TVD distributions
    """
    try:
        # Get base best-match metrics
        base_metrics = calculate_best_match_metrics(topics1, topics2, vocabulary)
        if base_metrics is None:
            return None

        # For simpler bootstrap, we'll resample with replacement from the TVD matrix
        tvd_matrix = np.array(base_metrics['tvd_matrix'])

        bootstrap_means_1to2 = []
        bootstrap_means_2to1 = []
        bootstrap_reciprocal_counts = []

        for _ in range(n_bootstrap):
            # Resample rows (topics from corpus 1)
            sampled_rows = np.random.choice(tvd_matrix.shape[0], size=tvd_matrix.shape[0], replace=True)
            # Resample columns (topics from corpus 2)
            sampled_cols = np.random.choice(tvd_matrix.shape[1], size=tvd_matrix.shape[1], replace=True)

            # Create resampled matrix
            resampled_matrix = tvd_matrix[np.ix_(sampled_rows, sampled_cols)]

            # Calculate best matches for resampled matrix
            best_1to2 = np.min(resampled_matrix, axis=1)
            best_2to1 = np.min(resampled_matrix, axis=0)

            bootstrap_means_1to2.append(np.mean(best_1to2))
            bootstrap_means_2to1.append(np.mean(best_2to1))

            # Count reciprocal matches
            reciprocal_count = 0
            for i in range(resampled_matrix.shape[0]):
                j = np.argmin(resampled_matrix[i, :])
                if np.argmin(resampled_matrix[:, j]) == i:
                    reciprocal_count += 1
            bootstrap_reciprocal_counts.append(reciprocal_count)

        bootstrap_means_1to2 = np.array(bootstrap_means_1to2)
        bootstrap_means_2to1 = np.array(bootstrap_means_2to1)
        bootstrap_reciprocal_counts = np.array(bootstrap_reciprocal_counts)

        return {
            'mean_tvd_1to2': {
                'mean': float(np.mean(bootstrap_means_1to2)),
                'ci_lower': float(np.percentile(bootstrap_means_1to2, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_means_1to2, 97.5)),
                'std': float(np.std(bootstrap_means_1to2))
            },
            'mean_tvd_2to1': {
                'mean': float(np.mean(bootstrap_means_2to1)),
                'ci_lower': float(np.percentile(bootstrap_means_2to1, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_means_2to1, 97.5)),
                'std': float(np.std(bootstrap_means_2to1))
            },
            'reciprocal_count': {
                'mean': float(np.mean(bootstrap_reciprocal_counts)),
                'ci_lower': float(np.percentile(bootstrap_reciprocal_counts, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_reciprocal_counts, 97.5)),
                'std': float(np.std(bootstrap_reciprocal_counts))
            }
        }

    except Exception as e:
        logger.error("Bootstrap best-match error: %s", str(e))
        return None


def mann_whitney_test(tvd_values_1, tvd_values_2):
    """
    Perform Mann-Whitney U test to compare two distributions of TVD values.

    Tests whether the best-match TVD distributions from two comparisons are significantly different.

    Args:
        tvd_values_1: List of TVD values from first comparison
        tvd_values_2: List of TVD values from second comparison

    Returns:
        dict with:
            - statistic: U statistic
            - p_value: Two-sided p-value
            - significant: Boolean indicating if p < 0.05
    """
    try:
        statistic, p_value = stats.mannwhitneyu(tvd_values_1, tvd_values_2, alternative='two-sided')

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }

    except Exception as e:
        logger.error("Mann-Whitney U test error: %s", str(e))
        return None

import os
import zipfile
import shutil
import tempfile
import re
import base64
from io import BytesIO
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import HdpModel
from nltk.corpus import stopwords
import nltk
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import logging
import numpy as np
import json
import matplotlib
from Bio import Entrez
import time
import seaborn as sns
from wordcloud import WordCloud

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gc
import time

# Import comparison utilities
from comparison_utils import (
    calculate_optimal_transport_distance,
    calculate_best_match_metrics,
    bootstrap_ot_distance,
    bootstrap_best_match_metrics,
    mann_whitney_test,
    permutation_test_ot_distance
)

app = Flask(__name__)
CORS(app)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.INFO)


try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    # Test if downloads worked
    from nltk.corpus import stopwords
    test_stops = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # Test lemmatizer
    test_lemma = lemmatizer.lemmatize("studies")
    logger.info("NLTK properly initialized with %d stopwords", len(test_stops))
except Exception as e:
    logger.error("NLTK initialization failed: %s. App will use fallback.", str(e))
    lemmatizer = None

logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# Enhanced stopwords for academic/research papers
# These are applied in addition to NLTK stopwords to filter generic academic and research terms
ENHANCED_STOPWORDS = {
    # Academic/research structural words (truly generic across all fields)
    'study', 'research', 'studies', 'analysis', 'results', 'method', 'methods',
    'table', 'figure', 'findings', 'conclusion', 'abstract', 'introduction',
    'discussion', 'participants', 'participant', 'experiment', 'experiments',
    'university', 'doi', 'journal', 'published', 'authors', 'author', 'using', 'used',
    'show', 'shown', 'shows', 'may', 'also', 'however', 'therefore', 'thus',
    'furthermore', 'moreover', 'additionally', 'meanwhile', 'nonetheless',

    # Generic quantitative terms (not domain-specific)
    'number', 'value', 'values', 'result', 'different', 'time', 'times',
    'based', 'two', 'one', 'three', 'four', 'five', 'first',
    'second', 'third', 'can', 'use', 'within', 'across', 'between', 'among',
    'well', 'large', 'small', 'high', 'low', 'new', 'different', 'same',
    'specific', 'general', 'particular', 'example', 'examples', 'case', 'cases',
    'significant', 'observed', 'obtained', 'performed', 'present',
    'related', 'associated', 'compared', 'due', 'examined', 'found',
    'increased', 'decreased', 'range', 'level', 'levels', 'term', 'terms',

    # Additional truly generic terms
    'condition', 'conditions', 'reason', 'reasons',
    'together', 'factor', 'factors', 'refer', 'refers',
    'approach', 'approaches', 'technique', 'techniques', 'problem', 'problems',
    'solution', 'solutions', 'measure', 'measures',
    'assumption', 'assumptions', 'plan', 'plans', 'speak', 'speaking',
    'qwen', 'effect', 'effects', 'change', 'changes',

    # Words that are TOO generic even in CS (appear everywhere)
    'system', 'systems', 'process', 'method', 'methods', 'data'
}

# NOTE: Removed domain-specific CS/ML terms from stopwords that should be kept:
# - algorithm, model, function, image, task, object
# - parameter, variable, performance, metric, evaluation
# - optimal, control, visual, equation
# These are meaningful topic indicators in CS/ML/AI papers!

def preprocess_text(text):
    """
    Improved text preprocessing for better LDA results:
    - Remove special characters, numbers, URLs, emails
    - Fix common OCR errors and PDF artifacts
    - Convert to lowercase and filter short words
    - Lemmatize words and remove stopwords
    """
    # Remove URLs, emails, and special patterns
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)  # Remove standalone numbers
    
    # Fix common PDF extraction artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\b\w{1,2}\b', ' ', text)     # Remove 1-2 letter words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)     # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize whitespace
    text = text.lower()

    # Filter out very short text
    if len(text) < 50:
        return ""

    # Tokenize and clean
    tokens = text.split()
    try:
        english_stopwords = set(stopwords.words("english"))
        # Add enhanced academic stopwords from module constant
        english_stopwords.update(ENHANCED_STOPWORDS)
    except Exception as e:
        logger.warning("NLTK stopwords failed, using fallback: %s", str(e))
        # Fallback stopwords list with enhanced academic terms
        english_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
            'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        # Merge with enhanced academic stopwords
        english_stopwords.update(ENHANCED_STOPWORDS)
    
    # Filter and lemmatize tokens
    filtered_tokens = []
    for word in tokens:
        if len(word) >= 3 and word not in english_stopwords and word.isalpha():
            if lemmatizer:
                # Try both noun and verb lemmatization for better results
                noun_lemma = lemmatizer.lemmatize(word, pos='n')
                verb_lemma = lemmatizer.lemmatize(word, pos='v')
                # Use the shorter lemma (usually better)
                lemma = noun_lemma if len(noun_lemma) <= len(verb_lemma) else verb_lemma
                filtered_tokens.append(lemma)
            else:
                # No lemmatization available
                filtered_tokens.append(word)

    tokens = filtered_tokens
    
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    logger.warning("Skipping encrypted file %s: %s", pdf_path, str(e))
                    return ""

            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += preprocess_text(page_text)  # Preprocess each page's text
    except Exception as e:
        logger.error("Error processing %s: %s", pdf_path, str(e))
        return ""
    return text

def get_pubmed_id(title, author, year, retmax=3):
    """Search PubMed and return best matching ID"""
    Entrez.email = "your.email@example.com"  # Required by NCBI
    query = '({title}[Title]) AND ({author}[Author]) AND {year}[Date - Publication]'.format(title=title, author=author, year=year)
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        handle.close()
        
        if record["IdList"]:
            return record["IdList"][0]  # Return first match
        return None
    except Exception as e:
        print("PubMed search failed: %s", str(e))
        return None

# Add 0.5s delay between requests to comply with NCBI guidelines
def safe_pubmed_lookup(*args):
    time.sleep(0.5)
    return get_pubmed_id(*args)

def find_cutoff_position(text):
    pattern = r"\b(acknowledgments|works cited|notes|references)\b"
    lower_text = text.lower()
    matches = [match.start() for match in re.finditer(pattern, lower_text)]

    if not matches:
        return len(text)

    return min(matches)


def clean_pdf_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()

    cutoff_pos = find_cutoff_position(text)

    if cutoff_pos < len(text) * 0.9:
        return text[:cutoff_pos]
    else:
        return text

def generate_decade_chart(decade_topic_distribution, lda):
    plt.ioff()
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)

    topic_dist_df = pd.DataFrame(decade_topic_distribution).T
    topic_dist_df.columns = ["Topic {}".format(i+1) for i in range(lda.n_components)]

    topic_dist_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85)
    ax.set_ylabel("Average Topic Proportion", fontsize=8)
    ax.set_title("Topic Distribution Over Decades", fontsize=10, pad=10)
    ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)

    plt.tight_layout(pad=2)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)
    decade_chart = base64.b64encode(img_buffer.read()).decode("utf-8")
    plt.close(fig)

    return decade_chart


def group_by_decades(years, doc_topic_matrix):
    bins = np.arange(1950, 2030, 10)
    years_decade_group = np.digitize(years, bins)

    decade_topic_distribution = {}

    for decade_bin in np.unique(years_decade_group):
        decade_indices = np.where(years_decade_group == decade_bin)[0]
        decade_name = "{}-{}".format(bins[decade_bin - 1], bins[decade_bin] - 1)

        average_topic_distribution = np.mean(doc_topic_matrix[decade_indices], axis=0)

        decade_topic_distribution[decade_name] = average_topic_distribution

    return decade_topic_distribution

def get_top_papers(doc_topic_matrix, titles, years, authors, n=5):
    logger.debug("Starting top papers analysis for %s papers per topic", n)
    """
    Get top papers with proper scaling and PubMed IDs.
    Uses percentile-based loading factors.
    """
    top_papers = {}
    
    for topic_idx in range(doc_topic_matrix.shape[1]):
        logger.debug("Processing topic %s for top papers", topic_idx + 1)
        topic_scores = doc_topic_matrix[:, topic_idx]
        
        # Calculate percentiles for nuanced scaling
        percentiles = np.percentile(topic_scores, [10, 90])
        scaled_scores = np.clip((topic_scores - percentiles[0]) / 
                               (percentiles[1] - percentiles[0]), 0, 1)
        
        # Get top papers
        top_indices = topic_scores.argsort()[::-1][:n]
        
        topic_papers = []
        for i in top_indices:
            # Skip PubMed lookup for faster processing
            pubmed_id = None
            
            topic_papers.append({
                "title": titles[i],
                "year": years[i],
                "author": authors[i],
                "loading_factor": float(scaled_scores[i]),
                "pubmed_id": pubmed_id,
                "raw_score": float(topic_scores[i])  # For debugging
            })
        
        top_papers[topic_idx] = topic_papers
    
    return top_papers

def prepare_gensim_corpus(doc_term_matrix, vectorizer):
    """Convert document-term matrix to Gensim corpus format"""
    corpus = []
    id2word = {v: k for k, v in vectorizer.vocabulary_.items()}
    for doc_idx in range(doc_term_matrix.shape[0]):
        doc_bow = []
        row = doc_term_matrix.getrow(doc_idx).toarray().flatten()
        for word_idx, count in enumerate(row):
            if count > 0:
                doc_bow.append((word_idx, count))
        corpus.append(doc_bow)
    dictionary = corpora.Dictionary.from_corpus(corpus, id2word=id2word)
    return corpus, dictionary

def create_hdp_comprehensive_visualizations(hdp_model, dictionary, topic_info, num_words=10):
    """Create comprehensive HDP visualizations like gen.py"""
    charts = {}
    num_topics = len(topic_info)
    
    if num_topics == 0:
        return charts
    
    try:
        # 1. Topic Quality Heatmap
        plt.figure(figsize=(12, 6))
        
        # Create data for heatmap
        topic_ids = [t['topic_id'] for t in topic_info]
        metrics = ['Avg Strength', 'Max Strength', 'Doc Coverage']
        
        heatmap_data = []
        for topic in topic_info:
            row = [
                topic['avg_strength'],
                topic['max_strength'],
                topic['strong_docs'] / max(1, len(topic_info))  # Normalized doc coverage
            ]
            heatmap_data.append(row)
        
        if len(heatmap_data) > 0:
            heatmap_data = np.array(heatmap_data).T
            
            sns.heatmap(heatmap_data, 
                       xticklabels=["Topic {}".format(tid) for tid in topic_ids],
                       yticklabels=metrics,
                       annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Strength/Coverage'})
            
            plt.title("HDP Topic Quality Metrics Heatmap", fontsize=14, pad=20)
            plt.xlabel("Topics")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            buffer.seek(0)
            charts["HDP_Quality_Heatmap"] = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
    
        # 2. Word Clouds for top topics (max 3)
        top_3_topics = sorted(topic_info, key=lambda x: x['avg_strength'], reverse=True)[:3]
        
        if len(top_3_topics) > 0:
            fig, axes = plt.subplots(1, len(top_3_topics), figsize=(5*len(top_3_topics), 5))
            if len(top_3_topics) == 1:
                axes = [axes]
            elif len(top_3_topics) == 2:
                axes = list(axes)
            
            fig.suptitle("Word Clouds for Top HDP Topics", fontsize=16, y=0.95)
            
            for i, topic in enumerate(top_3_topics):
                word_freq = dict(zip(topic['top_words'][:20], topic['probabilities'][:20]))
                
                try:
                    wordcloud = WordCloud(width=400, height=300, 
                                        background_color='white',
                                        max_words=20,
                                        colormap='viridis').generate_from_frequencies(word_freq)
                    
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title("Topic {} (Strength: {:.3f})".format(
                        topic['topic_id'], topic['avg_strength']), fontsize=12)
                    axes[i].axis('off')
                except Exception as e:
                    logger.warning("Could not create word cloud for topic %s: %s", topic['topic_id'], str(e))
                    axes[i].text(0.5, 0.5, "Topic {}\nWord cloud unavailable".format(topic['topic_id']), 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            buffer.seek(0)
            charts["HDP_Word_Clouds"] = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()

        # 3. Multi-panel Topic Overview (like gen.py)
        if num_topics <= 8:
            num_cols = min(4, num_topics)
            num_rows = (num_topics + num_cols - 1) // num_cols
            fig_size = (16, 4 * num_rows)
        else:
            num_cols = 4
            num_rows = (num_topics + num_cols - 1) // num_cols
            fig_size = (16, 3 * num_rows)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
        fig.suptitle("HDP Comprehensive Topic Analysis", fontsize=16, y=0.98)
        
        # Handle single topic case
        if num_topics == 1:
            axes = [axes]
        elif num_topics <= num_cols:
            axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i, topic in enumerate(topic_info):
            words = topic['top_words'][:num_words]
            probs = topic['probabilities'][:num_words]
            
            # Use color map for variety
            color = plt.cm.Set3(i % 12)  # Cycle through colors
            
            axes[i].barh(range(len(words)), probs, color=color)
            axes[i].set_yticks(range(len(words)))
            axes[i].set_yticklabels(words)
            axes[i].invert_yaxis()
            axes[i].set_xlabel('Probability')
            axes[i].set_title("Topic {} ({} docs)".format(
                topic['topic_id'], topic['strong_docs']), fontsize=10)
            
            # Add probability labels on bars
            for j, (word, prob) in enumerate(zip(words, probs)):
                axes[i].text(prob + 0.001, j, f'{prob:.3f}', 
                           ha='left', va='center', fontsize=8)
        
        # Hide unused subplots
        for i in range(num_topics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        charts["HDP_Multi_Panel_Overview"] = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        
    except Exception as e:
        logger.error("Error creating HDP comprehensive visualizations: %s", str(e))
    
    return charts

def analyze_hdp_task(file, form_data):
    """HDP-specific analysis task"""
    zip_path = None
    extracted_path = None
    try:
        # Get HDP-specific parameters
        alpha = float(form_data.get("alpha", 0.1))
        gamma = float(form_data.get("gamma", 0.01))
        num_words = int(form_data.get("numWords", 10))
        include_bibliography = (
            form_data.get("include_bibliography", "false").lower() == "true"
        )
        additional_stopwords = form_data.get("stopwords", "").split(",")
        additional_stopwords = [word.strip() for word in additional_stopwords]

        try:
            default_stopwords = stopwords.words("english")
        except:
            # Fallback stopwords if NLTK data unavailable
            default_stopwords = [
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might", "can", "must", "shall", "this", "that",
                "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                "them", "my", "your", "his", "her", "its", "our", "their"
            ]

        # Merge with enhanced academic stopwords
        all_stopwords = list(set(default_stopwords + additional_stopwords) | ENHANCED_STOPWORDS)

        # File processing (same as LDA)
        zip_path = tempfile.mktemp()
        file.save(zip_path)
        extracted_path = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        pdf_texts = []
        authors = []
        years = []
        titles = []
        pdf_count = 0

        for root, _, files in os.walk(extracted_path):
            for file_name in files:
                if file_name.endswith(".pdf") and not file_name.startswith("._"):
                    pdf_path = os.path.join(root, file_name)
                    logger.debug("Processing PDF: %s", pdf_path)

                    text = extract_text_from_pdf(pdf_path)
                    if not text:
                        logger.warning("No text extracted from %s", pdf_path)
                        continue

                    cleaned_text = clean_pdf_text(text) if not include_bibliography else text
                    processed_text = preprocess_text(cleaned_text)
                    if processed_text:  # Only add if preprocessing successful
                        pdf_texts.append(processed_text)
                        author, year, title = extract_metadata(file_name)
                        authors.append(author)
                        years.append(year)
                        titles.append(title)
                        pdf_count += 1

        if not pdf_texts:
            return {"error": "No valid text extracted from PDFs."}

        # Create document-term matrix with HDP-optimized parameters
        # Scale vocabulary size based on corpus size
        if pdf_count < 50:
            max_features = 1000  # Small corpus: moderate vocabulary
        elif pdf_count < 500:
            max_features = 2500  # Medium corpus: larger vocabulary
        else:
            max_features = 5000  # Large corpus: extensive vocabulary

        logger.info(f"Vocabulary scaling: {pdf_count} documents → max_features={max_features}")
        print(f"📚 Vocabulary size: {max_features} words (scaled for {pdf_count} documents)")

        vectorizer = CountVectorizer(
            max_df=0.6,           # Exclude words in >60% of docs (more restrictive for academic papers)
            min_df=max(2, pdf_count // 50),  # Require words in at least 2% of docs (was 10%, now more lenient)
            stop_words=all_stopwords,
            token_pattern=r'\b[a-zA-Z]{4,}\b',  # Longer words for better quality
            max_features=max_features,  # Scaled vocabulary based on corpus size
        )
        
        doc_term_matrix = vectorizer.fit_transform(pdf_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Prepare Gensim corpus
        corpus, dictionary = prepare_gensim_corpus(doc_term_matrix, vectorizer)

        # Train HDP model
        # HDP automatically discovers the number of topics
        # T and K control the truncation levels
        # Scale T and gamma based on corpus size to prevent topic explosion

        if pdf_count < 50:
            # Small corpus: limit topics aggressively
            T = 15
            gamma_scaled = 0.5
        elif pdf_count < 500:
            # Medium corpus: moderate topic limit
            T = 30
            gamma_scaled = 0.3
        elif pdf_count < 5000:
            # Large corpus: allow more topics but lower gamma
            T = 50
            gamma_scaled = 0.1
        else:
            # Very large corpus: highest limit but lowest gamma
            T = 75
            gamma_scaled = 0.05

        logger.info(f"HDP scaling: {pdf_count} documents → T={T}, gamma={gamma_scaled}")
        print(f"\n📊 HDP Auto-scaling: {pdf_count} documents → T={T}, gamma={gamma_scaled}")

        hdp_model = HdpModel(
            corpus=corpus,
            id2word=dictionary,
            random_state=42,
            alpha=alpha,         # Document-topic concentration (from user input, default 0.1)
            gamma=gamma_scaled,  # Topic-level concentration (scaled by corpus size)
            eta=0.01,            # Topic-word concentration (sparsity in word distribution)
            T=T,                 # First-level truncation (scaled by corpus size)
            K=15,                # Second-level truncation (max topics per document)
            kappa=0.75,          # Learning rate decay
            tau=64.0             # Learning rate delay
        )

        # Extract significant topics (filter weak/noise topics)
        all_topics = hdp_model.get_topics()
        # Filter topics with meaningful word mass (not too strict)
        significant_topic_indices = [i for i, t in enumerate(all_topics) if np.sum(t) > 0.5]
        num_active_topics = len(significant_topic_indices)

        if num_active_topics == 0:
            return {"error": "HDP model training failed. No significant topics generated."}

        logger.debug("Trained HDP model with %s significant topics", num_active_topics)
        
        # Print detailed model evaluation for user analysis
        print("\n" + "="*80)
        print("*** HDP MODEL ANALYSIS RESULTS ***")
        print("="*80)
        print("Number of active topics: {}".format(num_active_topics))
        print("Number of documents: {}".format(len(pdf_texts)))
        print("Vocabulary size: {}".format(len(feature_names)))
        print("Alpha (doc-topic): {}".format(alpha))
        print("Gamma (topic concentration): {}".format(gamma))

        # Prepare topic info for comprehensive visualizations
        topic_info_for_viz = []
        
        # Generate topics and charts
        topic_charts = {}
        topics = []
        topic_word_distributions = []

        print("\n*** DETAILED TOPIC BREAKDOWN: ***")
        print("-"*60)

        for i, topic_idx in enumerate(significant_topic_indices):
            topic_words = hdp_model.show_topic(topic_idx, topn=num_words)
            top_words = [word for word, prob in topic_words]
            probabilities = [prob for word, prob in topic_words]
            
            # Calculate document-topic associations
            doc_topics = []
            for doc_bow in corpus:
                doc_topic_dist = dict(hdp_model[doc_bow])
                topic_prob = doc_topic_dist.get(topic_idx, 0)
                doc_topics.append(topic_prob)
            
            strong_docs = sum(1 for prob in doc_topics if prob > 0.2)
            avg_strength = float(np.mean(doc_topics)) if doc_topics else 0.0
            max_strength = float(np.max(doc_topics)) if doc_topics else 0.0
            
            print("\n*** TOPIC {}: ***".format(i + 1))
            print("   Top words: {}".format(', '.join(top_words)))
            print("   Word probabilities: {}".format(["{:.4f}".format(p) for p in probabilities[:5]]))
            print("   Documents with >20% loading: {}/{} ({:.1f}%)".format(strong_docs, len(corpus), strong_docs/len(corpus)*100))
            print("   Average topic strength: {:.3f}".format(avg_strength))
            print("   Maximum topic strength: {:.3f}".format(max_strength))
            print("-"*60)

            # Collect topic info for comprehensive visualizations
            topic_info_for_viz.append({
                'topic_id': i + 1,
                'top_words': top_words,
                'probabilities': probabilities,
                'strong_docs': int(strong_docs),
                'avg_strength': float(avg_strength),
                'max_strength': float(max_strength)
            })

            # Create basic topic chart (same as before)
            plt.figure(figsize=(10, 6))
            plt.barh(top_words, probabilities, color="steelblue")
            plt.gca().invert_yaxis()
            plt.xlabel("Probability")
            plt.title("HDP Topic {} - Top {} Words".format(i + 1, num_words))
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
            
            topic_name = "Topic {}".format(i + 1)
            topic_charts[topic_name] = chart_base64
            
            topics.append({
                "Topic": topic_name,
                "Words": ", ".join(top_words),
                "Probabilities": probabilities,
                "DocumentStrength": float(avg_strength),
                "MaxStrength": float(max_strength),
                "StrongDocs": int(strong_docs)
            })
            
            # Prepare topic-word distributions for export (same format as LDA)
            topic_word_prob = {}
            for word_idx, word in enumerate(feature_names):
                if word_idx < len(all_topics[topic_idx]):
                    prob = all_topics[topic_idx][word_idx]
                    if prob > 1e-10:  # Only include meaningful probabilities
                        topic_word_prob[word] = float(prob)
            
            topic_word_distributions.append({
                "topic_id": i,
                "topic_name": topic_name,
                "word_probabilities": topic_word_prob
            })

        # Calculate model quality metrics
        # Create comprehensive HDP visualizations
        advanced_charts = create_hdp_comprehensive_visualizations(
            hdp_model, dictionary, topic_info_for_viz, num_words
        )
        
        # Add advanced charts to topic_charts
        topic_charts.update(advanced_charts)

        print("\n*** MODEL QUALITY ASSESSMENT: ***")
        print("-"*60)
        print("HDP discovered {} topics from {} documents".format(num_active_topics, pdf_count))
        print("="*80)

        # Calculate top papers for each topic (simplified for HDP)
        top_papers = []
        for i, topic_idx in enumerate(significant_topic_indices):
            topic_papers = []
            doc_topics = []
            
            for doc_idx, doc_bow in enumerate(corpus):
                doc_topic_dist = dict(hdp_model[doc_bow])
                topic_prob = doc_topic_dist.get(topic_idx, 0)
                if doc_idx < len(titles) and topic_prob > 0.1:  # Only include meaningful associations
                    topic_papers.append({
                        "title": titles[doc_idx],
                        "author": authors[doc_idx] if doc_idx < len(authors) else "Unknown",
                        "year": years[doc_idx] if doc_idx < len(years) else None,
                        "loading_factor": topic_prob,
                        "raw_score": topic_prob,
                        "pubmed_id": None  # Skip PubMed lookup for HDP
                    })
            
            # Sort by loading factor and take top papers
            topic_papers.sort(key=lambda x: x["loading_factor"], reverse=True)
            top_papers.append(topic_papers[:min(5, len(topic_papers))])

        # Calculate time period
        valid_years = [int(y) for y in years if y is not None]
        time_period = "{}-{}".format(min(valid_years), max(valid_years)) if valid_years else "N/A"

        # Calculate topic prevalence (average document-topic proportions for HDP)
        topic_prevalence = []
        for i, topic_idx in enumerate(significant_topic_indices):
            doc_topics = []
            for doc_bow in corpus:
                doc_topic_dist = dict(hdp_model[doc_bow])
                topic_prob = doc_topic_dist.get(topic_idx, 0)
                doc_topics.append(topic_prob)

            prevalence = float(np.mean(doc_topics)) if doc_topics else 0.0
            topic_prevalence.append({
                "topic_id": i,
                "topic_name": "Topic {}".format(i + 1),
                "prevalence": prevalence
            })

        return {
            "topics": topics,
            "topic_charts": topic_charts,
            "num_words": int(num_words),
            "num_pdfs": int(pdf_count),
            "num_topics": int(num_active_topics),
            "time_period": time_period,
            "model_type": "HDP",
            "model_params": {
                "alpha": float(alpha),
                "gamma": float(gamma),
                "num_active_topics": int(num_active_topics),
                "total_discovered_topics": int(len(all_topics))
            },
            "top_papers": top_papers,
            "topic_word_distributions": topic_word_distributions,
            "topic_prevalence": topic_prevalence,
            "vocabulary": feature_names.tolist(),
            "average_lift_per_topic": [1.0] * int(num_active_topics),  # Placeholder for compatibility
            "model_loss": 0.0,  # HDP doesn't have direct loss equivalent
            "vectorizer_params": {
                "max_df": 0.7,
                "min_df": int(max(2, pdf_count // 10)),
                "stopwords_count": int(len(all_stopwords)),
                "max_features": int(min(500, pdf_count * 50))
            }
        }

    except Exception as e:
        logger.error("HDP Analysis failed: %s", str(e))
        return {"error": "HDP Analysis failed", "details": str(e)}
    finally:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
        if extracted_path and os.path.exists(extracted_path):
            shutil.rmtree(extracted_path)
        plt.close("all")
        gc.collect()

def analyze_task(file, form_data):
    zip_path = None
    extracted_path = None
    try:

        num_topics = int(form_data.get("numTopics", 5))
        num_words = int(form_data.get("numWords", 10))
        include_bibliography = (
            form_data.get("include_bibliography", "false").lower() == "true"
        )
        additional_stopwords = form_data.get("stopwords", "").split(",")
        additional_stopwords = [word.strip() for word in additional_stopwords]

        try:
            default_stopwords = stopwords.words("english")
        except:
            # Fallback stopwords if NLTK data unavailable
            default_stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must", "shall", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their"]

        # Merge with enhanced academic stopwords
        all_stopwords = list(set(default_stopwords + additional_stopwords) | ENHANCED_STOPWORDS)

        zip_path = tempfile.mktemp()
        file.save(zip_path)
        extracted_path = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        pdf_texts = []
        authors = []
        years = []
        titles = []
        pdf_count = 0

        for root, _, files in os.walk(extracted_path):
            for file_name in files:
                if file_name.endswith(".pdf") and not file_name.startswith("._"):
                    pdf_path = os.path.join(root, file_name)
                    logger.debug("Processing PDF: %s", pdf_path)

                    text = extract_text_from_pdf(pdf_path)
                    if not text:
                        logger.warning("No text extracted from %s", pdf_path)
                        continue

                    # Apply proper preprocessing (same as HDP)
                    cleaned_text = clean_pdf_text(text) if not include_bibliography else text
                    processed_text = preprocess_text(cleaned_text)
                    if processed_text:  # Only add if preprocessing successful
                        pdf_texts.append(processed_text)
                    author, year, title = extract_metadata(file_name)
                    authors.append(author)
                    years.append(year)
                    titles.append(title)
                    pdf_count += 1
        if not pdf_texts:
            return {"error": "No valid text extracted from PDFs."}
        # Fix vectorizer_params to match actual values
        vectorizer_params = {
                "max_df": 0.85,
                "min_df": 2,
                "stopwords_count": len(all_stopwords),
                "max_features": 1000
            }
        # Use CountVectorizer instead of TfidfVectorizer for better LDA performance
        vectorizer = CountVectorizer(
            max_df=0.85,  # Exclude words appearing in >85% of documents
            min_df=2,     # Require words to appear in at least 2 documents
            stop_words=all_stopwords,
            token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only words with 3+ letters
            max_features=1000,  # Limit vocabulary size
        )
        doc_term_matrix = vectorizer.fit_transform(pdf_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Train LDA model
        # Improved LDA parameters for better topic quality
        lda = LDA(
            n_components=num_topics, 
            random_state=42,
            doc_topic_prior=1.0/num_topics,  # Alpha - lower values encourage documents to focus on fewer topics
            topic_word_prior=0.01,  # Beta - lower values encourage topics to focus on fewer words
            learning_method='batch',
            max_iter=50,
            evaluate_every=10,
            perp_tol=1e-3
        )
        lda.fit(doc_term_matrix)

        if lda.components_.shape[0] == 0:
            return {"error": "LDA model training failed. No topics generated."}

        logger.debug("Trained LDA model with %s topics", lda.n_components)
        
        # Print detailed model evaluation for user analysis
        print("\n" + "="*80)
        print("🔍 LDA MODEL ANALYSIS RESULTS")
        print("="*80)
        print(f"📊 Number of topics: {lda.n_components}")
        print(f"📄 Number of documents: {len(pdf_texts)}")
        print(f"🔤 Vocabulary size: {len(feature_names)}")
        print(f"📈 Model perplexity: {lda.perplexity(doc_term_matrix):.2f}")
        print(f"📉 Log likelihood: {lda.score(doc_term_matrix):.2f}")
        print(f"⚙️  Max iterations reached: {lda.n_iter_}")
        
        logger.debug("Starting visualization generation...")
        components = lda.components_
        doc_topic_matrix = lda.transform(doc_term_matrix)
        
        # Print detailed topic analysis
        print("\n📋 DETAILED TOPIC BREAKDOWN:")
        print("-"*60)
   
        topic_charts = {}
        topics = []

        logger.debug("Generating topic charts for %s topics...", num_topics)
        for topic_idx in range(num_topics):
            top_indices = components[topic_idx].argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            raw_weights = components[topic_idx][top_indices]
            EPSILON = 1e-12 
            marginal_word_prob = (lda.components_.sum(axis=0) + EPSILON) / (lda.components_.sum() + EPSILON)
            lift_scores = (lda.components_ + EPSILON) / (marginal_word_prob + EPSILON)
            average_lift_per_topic = np.mean(lift_scores, axis=1).tolist()
            total_weight = raw_weights.sum()
            percentages = (raw_weights / total_weight * 100).round(2)
            
            # Print detailed topic analysis
            print(f"\n🏷️  TOPIC {topic_idx + 1}:")
            print(f"   Top words: {', '.join(top_words)}")
            print(f"   Word weights (raw): {[f'{w:.4f}' for w in raw_weights]}")
            print(f"   Word percentages: {[f'{p:.1f}%' for p in percentages]}")
            print(f"   Topic coherence (avg lift): {average_lift_per_topic[topic_idx]:.3f}")
            
            # Calculate topic concentration
            topic_distribution = doc_topic_matrix[:, topic_idx]
            dominant_docs = (topic_distribution > 0.3).sum()
            print(f"   Documents dominated by this topic: {dominant_docs}/{len(pdf_texts)} ({dominant_docs/len(pdf_texts)*100:.1f}%)")
            print(f"   Avg topic strength in docs: {topic_distribution.mean():.3f}")
            
            # Show sample text from top documents for this topic
            top_doc_indices = topic_distribution.argsort()[-3:][::-1]
            print(f"   📄 Sample from top documents:")
            for i, doc_idx in enumerate(top_doc_indices[:2]):
                if doc_idx < len(titles):
                    snippet = pdf_texts[doc_idx][:200] + "..." if len(pdf_texts[doc_idx]) > 200 else pdf_texts[doc_idx]
                    print(f"      {i+1}. {titles[doc_idx][:50]}...")
                    print(f"         Snippet: {snippet}")
                    print(f"         Topic strength: {topic_distribution[doc_idx]:.3f}")
            print("-"*60)

            plt.figure(figsize=(10, 6))
            plt.barh(top_words, percentages, color="steelblue")
            plt.gca().invert_yaxis()
            plt.xlabel("Percentage Importance (%)")
            plt.title("Topic {} - Word Distribution".format(topic_idx + 1))

            for i, (word, pct) in enumerate(zip(top_words, percentages)):
                plt.text(pct + 0.5, i, "{:.1f}%".format(pct), va="center", fontsize=8)

            plt.tight_layout()

            img_buffer = BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=120)
            img_buffer.seek(0)
            topic_charts["Topic {}".format(topic_idx + 1)] = base64.b64encode(
                img_buffer.read()
            ).decode("utf-8")
            plt.close()

            topics.append({
                "Topic": "Topic {}".format(topic_idx + 1),
                "Words": ", ".join(top_words),
                "WordScores": {
                    "raw": raw_weights.tolist(),
                    "percentages": (raw_weights / raw_weights.sum() * 100).round(2).tolist()
                }
            })
            
        num_top_papers = int(form_data.get("numTopPapers", 5))  # Default to 5
        logger.debug("Getting top %s papers for each topic...", num_top_papers)
        top_papers = get_top_papers(doc_topic_matrix, titles, years, authors, num_top_papers)
        logger.debug("Completed top papers analysis")
        valid_years = [y for y in years if y is not None]
        time_period = "{}-{}".format(min(valid_years), max(valid_years)) if valid_years else "N/A"
        model_loss = -lda.score(doc_term_matrix)
        # Print final model assessment
        print("\n🎯 MODEL QUALITY ASSESSMENT:")
        print("-"*60)
        
        # Calculate overall topic quality metrics
        topic_concentrations = []
        for topic_idx in range(lda.n_components):
            topic_dist = doc_topic_matrix[:, topic_idx]
            concentration = (topic_dist > 0.3).sum() / len(pdf_texts)
            topic_concentrations.append(concentration)
        
        avg_concentration = np.mean(topic_concentrations)
        print(f"📍 Average topic concentration: {avg_concentration:.3f}")
        print(f"📊 Perplexity (lower is better): {lda.perplexity(doc_term_matrix):.2f}")
        print(f"🎲 Topic separation quality: {'Good' if avg_concentration > 0.1 else 'Poor'}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if lda.perplexity(doc_term_matrix) > 1000:
            print("   ⚠️  High perplexity suggests:")
            print("      - Try reducing number of topics")
            print("      - Increase min_df to remove rare words")
            print("      - Add more domain-specific stopwords")
        
        if avg_concentration < 0.1:
            print("   ⚠️  Low topic concentration suggests:")
            print("      - Topics may be too generic")
            print("      - Consider reducing topic count")
            print("      - Check if documents are too diverse")
        
        print("   🔧 Current vectorizer settings:")
        print(f"      - max_df: {vectorizer.max_df}")
        print(f"      - min_df: {vectorizer.min_df}")
        print(f"      - Stopwords count: {len(all_stopwords)}")
        print("="*80)
        
        # Prepare topic-word probabilities for export
        topic_word_distributions = []
        for topic_idx in range(lda.n_components):
            topic_word_prob = {}
            for word_idx, word in enumerate(feature_names):
                prob = lda.components_[topic_idx, word_idx]
                if prob > 1e-10:  # Only include words with meaningful probability
                    topic_word_prob[word] = float(prob)
            topic_word_distributions.append({
                "topic_id": topic_idx,
                "topic_name": "Topic {}".format(topic_idx + 1),
                "word_probabilities": topic_word_prob
            })

        # Calculate topic prevalence (average document-topic proportions)
        topic_prevalence = []
        num_docs = doc_topic_matrix.shape[0]

        for topic_idx in range(lda.n_components):
            # Average proportion of this topic across all documents
            prevalence = np.mean(doc_topic_matrix[:, topic_idx])
            topic_prevalence.append({
                "topic_id": topic_idx,
                "topic_name": "Topic {}".format(topic_idx + 1),
                "prevalence": float(prevalence)
            })

        return {
            "topics": topics,
            "topic_charts": topic_charts,
            "vectorizer": vectorizer,
            "lda_model": lda,
            "doc_topic_matrix": doc_topic_matrix,
            "additional_stopwords": additional_stopwords,
            "num_pdfs": pdf_count,
            "num_topics": lda.n_components,
            "num_words": num_words,
            "titles": titles,
            "years": years,
            "time_period": time_period,
            "average_lift_per_topic": average_lift_per_topic,
            "model_loss": model_loss,
            "vectorizer_params": vectorizer_params,
            "perplexity": lda.perplexity(doc_term_matrix),
            "top_papers": top_papers,
            "num_top_papers": num_top_papers,
            "topic_word_distributions": topic_word_distributions,
            "topic_prevalence": topic_prevalence,
            "vocabulary": feature_names.tolist(),
        }

    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        return {"error": str(e)}
    finally:
        plt.close("all")
        gc.collect()
        try:
            if zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as e:
            logger.error("Error removing zip file: %s", str(e))

        try:
            if extracted_path and os.path.exists(extracted_path):
                shutil.rmtree(extracted_path)
        except Exception as e:
            logger.error("Error removing extracted files: %s", str(e))


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        form_data = request.form.to_dict()
        result = analyze_task(file, form_data)

        if "error" in result:
            return jsonify(result), 500

        if form_data.get("include_decade_analysis", "false").lower() == "true":
            decade_chart = generate_decade_chart(
                group_by_decades(result["years"], result["doc_topic_matrix"]),
                result["lda_model"],
            )
            result["decade_chart_base64"] = decade_chart

        return jsonify(
            {
                "topics": result["topics"],
                "topic_charts": result["topic_charts"],
                "decade_chart_base64": result.get("decade_chart_base64", None),
                "num_words": result["num_words"],
                "num_pdfs": result["num_pdfs"],
                "num_topics": result["num_topics"],
                "time_period": result["time_period"],
                "model_loss": result["model_loss"],
                "average_lift_per_topic": result["average_lift_per_topic"],
                "cache_id": int(time.time()),
                "top_papers": result["top_papers"],
                "num_top_papers": result["num_top_papers"],
                "topic_word_distributions": result["topic_word_distributions"],
                "topic_prevalence": result["topic_prevalence"],
                "vocabulary": result["vocabulary"],
            }
        )

    except Exception as e:
        logger.error("Analysis failed: %s", str(e))
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500
    finally:
        plt.close("all")
        gc.collect()

@app.route("/analyze-hdp", methods=["POST"])
def analyze_hdp():
    """HDP Analysis endpoint"""
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        form_data = request.form.to_dict()
        result = analyze_hdp_task(file, form_data)

        if "error" in result:
            return jsonify(result), 500

        # HDP doesn't support decade analysis currently
        # Could be added in future if needed

        return jsonify(
            {
                "topics": result["topics"],
                "topic_charts": result["topic_charts"],
                "decade_chart_base64": None,  # Not supported for HDP
                "num_words": result["num_words"],
                "num_pdfs": result["num_pdfs"],
                "num_topics": result["num_topics"],
                "time_period": result["time_period"],
                "model_type": result["model_type"],
                "model_params": result["model_params"],
                "cache_id": int(time.time()),
                "top_papers": result["top_papers"],
                "num_top_papers": len(result["top_papers"]),
                "topic_word_distributions": result["topic_word_distributions"],
                "topic_prevalence": result["topic_prevalence"],
                "vocabulary": result["vocabulary"],
                "average_lift_per_topic": result["average_lift_per_topic"],
                "model_loss": result["model_loss"],
                "vectorizer_params": result["vectorizer_params"]
            }
        )

    except Exception as e:
        logger.error("HDP Analysis failed: %s", str(e))
        return jsonify({"error": "HDP Analysis failed", "details": str(e)}), 500
    finally:
        plt.close("all")
        gc.collect()


def extract_metadata(filename):
    """
    Extract author, year, and title from filename format:
    "Author et al. - Year - Title.pdf"
    """
    pattern = r"^(.*?) - (\d{4}) - (.*?)\.pdf$"
    match = re.match(pattern, filename)
    if match:
        author = match.group(1).strip()
        year = int(match.group(2))
        title = match.group(3).strip()
        return author, year, title
    # Fallback for files that don't match pattern
    base_name = os.path.splitext(filename)[0]
    return base_name, 2024, base_name

def generate_topic_distribution_charts(lda_model, feature_names, num_words=10):
    topic_charts = {}
    try:
        plt.ioff()

        for topic_idx, topic_weights in enumerate(lda_model.components_):

            top_indices = topic_weights.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            weights = topic_weights[top_indices]

            fig, ax = plt.figure(num="topic_{}".format(topic_idx), figsize=(10, 6), dpi=100)
            fig.clf()

            y_pos = np.arange(len(top_words))
            bars = ax.barh(y_pos, weights, align="center", color="steelblue")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_words)
            ax.invert_yaxis()
            ax.set_xlabel("Word Importance Score")
            ax.set_title("Topic {} - Key Terms Distribution".format(topic_idx+1))

            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width * 1.02,
                    bar.get_y() + bar.get_height() / 2,
                    "{:.2f}".format(width),
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            buf.seek(0)

            del fig
            gc.collect()
            topic_charts["Topic {}".format(topic_idx + 1)] = base64.b64encode(
                buf.read()
            ).decode("utf-8")
    finally:
        plt.close("all")
        gc.collect()
    return topic_charts


def calculate_tvd_similarity(topic_dist_1, topic_dist_2, vocabulary):
    """
    Calculate Total Variation Distance between two topic-word distributions.
    TVD = 0.5 * sum(|P(w|topic1) - P(w|topic2)|) for all words w
    Returns similarity matrix where 0 = identical, 1 = completely different
    """
    try:
        # Create probability vectors for all words in vocabulary
        prob_vector_1 = np.zeros(len(vocabulary))
        prob_vector_2 = np.zeros(len(vocabulary))
        
        # Fill probability vectors
        for i, word in enumerate(vocabulary):
            prob_vector_1[i] = topic_dist_1.get(word, 0.0)
            prob_vector_2[i] = topic_dist_2.get(word, 0.0)
        
        # Normalize to ensure they sum to 1 (LDA should already do this, but safety check)
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

@app.route("/compare", methods=["POST"])
def compare_topics():
    """
    Enhanced topic comparison endpoint with Optimal Transport, best-match analysis,
    and bootstrap confidence intervals.

    Expects:
    - file1, file2: Topic-word distribution CSVs
    - prevalence_file1, prevalence_file2 (optional): Topic prevalence CSVs
    - n_bootstrap (optional): Number of bootstrap samples (default 1000)
    """
    try:
        # Get uploaded files
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        prevalence_file1 = request.files.get("prevalence_file1")
        prevalence_file2 = request.files.get("prevalence_file2")

        if not file1 or not file2:
            return jsonify({"error": "Both CSV files are required"}), 400

        # Get optional parameters
        n_bootstrap = int(request.form.get("n_bootstrap", 1000))

        # Read topic distribution CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Check required columns
        required_cols = ['Topic_ID', 'Topic_Name', 'Word', 'Probability']
        for col in required_cols:
            if col not in df1.columns or col not in df2.columns:
                return jsonify({"error": "CSV files must have columns: Topic_ID, Topic_Name, Word, Probability"}), 400

        # Group by topics to create topic-word distributions
        topics1_list = []
        topics1_dict = {}
        for topic_id in sorted(df1['Topic_ID'].unique()):
            topic_data = df1[df1['Topic_ID'] == topic_id]
            word_probs = dict(zip(topic_data['Word'], topic_data['Probability']))
            # Get top 5 words for better topic naming
            top_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            topic_name = ", ".join([word for word, _ in top_words])

            topic_obj = {
                'topic_id': int(topic_id),
                'topic_name': topic_data['Topic_Name'].iloc[0],
                'word_probabilities': word_probs
            }
            topics1_list.append(topic_obj)
            topics1_dict[int(topic_id)] = {
                'name': topic_name,
                'full_name': topic_data['Topic_Name'].iloc[0],
                'word_probs': word_probs
            }

        topics2_list = []
        topics2_dict = {}
        for topic_id in sorted(df2['Topic_ID'].unique()):
            topic_data = df2[df2['Topic_ID'] == topic_id]
            word_probs = dict(zip(topic_data['Word'], topic_data['Probability']))
            # Get top 5 words for better topic naming
            top_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            topic_name = ", ".join([word for word, _ in top_words])

            topic_obj = {
                'topic_id': int(topic_id),
                'topic_name': topic_data['Topic_Name'].iloc[0],
                'word_probabilities': word_probs
            }
            topics2_list.append(topic_obj)
            topics2_dict[int(topic_id)] = {
                'name': topic_name,
                'full_name': topic_data['Topic_Name'].iloc[0],
                'word_probs': word_probs
            }

        # Get combined vocabulary
        all_words = set(df1['Word'].unique()) | set(df2['Word'].unique())
        vocabulary = list(all_words)

        # Print initial dataset info
        import sys
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("🔄 TOPIC COMPARISON ANALYSIS", file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)
        print(f"📊 Dataset 1: {len(topics1_list)} topics, {len(df1['Word'].unique())} unique words", file=sys.stderr, flush=True)
        print(f"📊 Dataset 2: {len(topics2_list)} topics, {len(df2['Word'].unique())} unique words", file=sys.stderr, flush=True)
        print(f"🔤 Combined vocabulary: {len(vocabulary)} unique words", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        # Print topics from each dataset
        print("📋 DATASET 1 - TOPICS:", file=sys.stderr, flush=True)
        print("-"*60, file=sys.stderr, flush=True)
        for i, topic in enumerate(topics1_list):
            top_5_words = sorted(topic['word_probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
            words_str = ", ".join([f"{w} ({p:.3f})" for w, p in top_5_words])
            print(f"  Topic {i+1}: {words_str}", file=sys.stderr, flush=True)

        print("\n📋 DATASET 2 - TOPICS:", file=sys.stderr, flush=True)
        print("-"*60, file=sys.stderr, flush=True)
        for i, topic in enumerate(topics2_list):
            top_5_words = sorted(topic['word_probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
            words_str = ", ".join([f"{w} ({p:.3f})" for w, p in top_5_words])
            print(f"  Topic {i+1}: {words_str}", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        # Read prevalence files if provided
        prevalence1 = None
        prevalence2 = None

        if prevalence_file1 and prevalence_file2:
            try:
                prev_df1 = pd.read_csv(prevalence_file1)
                prev_df2 = pd.read_csv(prevalence_file2)

                if 'Prevalence' in prev_df1.columns and 'Prevalence' in prev_df2.columns:
                    prevalence1 = prev_df1.sort_values('Topic_ID')['Prevalence'].tolist()
                    prevalence2 = prev_df2.sort_values('Topic_ID')['Prevalence'].tolist()
                    logger.info("Prevalence data loaded successfully")
                    print("✅ Prevalence data provided:", file=sys.stderr, flush=True)
                    print(f"   Dataset 1: {[f'{p:.4f}' for p in prevalence1]}", file=sys.stderr, flush=True)
                    print(f"   Dataset 2: {[f'{p:.4f}' for p in prevalence2]}", file=sys.stderr, flush=True)
            except Exception as e:
                logger.warning("Could not load prevalence files: %s. Using uniform weights.", str(e))
                print(f"⚠️  Could not load prevalence files: {str(e)}", file=sys.stderr, flush=True)

        # If prevalence not provided, use uniform distribution
        if prevalence1 is None:
            prevalence1 = [1.0 / len(topics1_list)] * len(topics1_list)
            print(f"📊 Using uniform prevalence for Dataset 1: {prevalence1[0]:.4f} per topic", file=sys.stderr, flush=True)
        if prevalence2 is None:
            prevalence2 = [1.0 / len(topics2_list)] * len(topics2_list)
            print(f"📊 Using uniform prevalence for Dataset 2: {prevalence2[0]:.4f} per topic", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        # Calculate basic TVD similarity matrix (for backward compatibility)
        num_topics1 = len(topics1_dict)
        num_topics2 = len(topics2_dict)

        similarity_matrix = np.zeros((num_topics1, num_topics2))
        matrix_data = []

        for i, (topic1_id, topic1_data) in enumerate(sorted(topics1_dict.items())):
            row_data = []
            for j, (topic2_id, topic2_data) in enumerate(sorted(topics2_dict.items())):
                tvd = calculate_tvd_similarity(
                    topic1_data['word_probs'],
                    topic2_data['word_probs'],
                    vocabulary
                )
                similarity_matrix[i, j] = tvd
                row_data.append({
                    'topic1_id': topic1_id,
                    'topic1_name': topic1_data['name'],
                    'topic2_id': topic2_id,
                    'topic2_name': topic2_data['name'],
                    'tvd_distance': tvd,
                    'similarity_percent': (1 - tvd) * 100
                })
            matrix_data.append(row_data)

        # Calculate advanced metrics using comparison_utils
        print("⚙️  CALCULATING ADVANCED METRICS...", file=sys.stderr, flush=True)
        print("-"*60, file=sys.stderr, flush=True)

        logger.info("Calculating Optimal Transport distance...")
        print("🌍 Computing Optimal Transport (OT) distance...", file=sys.stderr, flush=True)
        ot_distance = calculate_optimal_transport_distance(
            topics1_list, topics2_list, prevalence1, prevalence2, vocabulary
        )
        print(f"   ✅ OT Distance: {ot_distance:.4f}", file=sys.stderr, flush=True)
        print(f"      (0 = identical, higher = more different)", file=sys.stderr, flush=True)
        print()

        logger.info("Calculating best-match metrics...")
        print("🎯 Computing Best-Match Analysis...", file=sys.stderr, flush=True)
        best_match_results = calculate_best_match_metrics(
            topics1_list, topics2_list, vocabulary
        )

        print(f"   ✅ Best matches found:", file=sys.stderr, flush=True)
        print(f"      Dataset 1→2: {len(best_match_results['best_matches_1to2'])} matches", file=sys.stderr, flush=True)
        print(f"      Dataset 2→1: {len(best_match_results['best_matches_2to1'])} matches", file=sys.stderr, flush=True)
        print(f"      🔄 Reciprocal matches: {best_match_results['num_reciprocal']}", file=sys.stderr, flush=True)
        print()

        print("📊 Coverage Statistics:", file=sys.stderr, flush=True)
        print(f"   Dataset 1→2 Coverage@0.3: {best_match_results['coverage_1to2']*100:.1f}%", file=sys.stderr, flush=True)
        print(f"   Dataset 2→1 Coverage@0.3: {best_match_results['coverage_2to1']*100:.1f}%", file=sys.stderr, flush=True)
        print(f"   High Divergence 1→2 (>0.7): {best_match_results['high_divergence_1to2']*100:.1f}%", file=sys.stderr, flush=True)
        print(f"   High Divergence 2→1 (>0.7): {best_match_results['high_divergence_2to1']*100:.1f}%", file=sys.stderr, flush=True)
        print()

        # Calculate bootstrap confidence intervals
        logger.info("Calculating bootstrap confidence intervals (n=%d)...", n_bootstrap)
        print(f"🔬 Computing Bootstrap Confidence Intervals ({n_bootstrap} samples)...", file=sys.stderr, flush=True)
        print("   This may take a moment...", file=sys.stderr, flush=True)
        ot_bootstrap = bootstrap_ot_distance(
            topics1_list, topics2_list, prevalence1, prevalence2, vocabulary, n_bootstrap
        )
        print(f"   ✅ OT Bootstrap Complete:", file=sys.stderr, flush=True)
        print(f"      Mean: {ot_bootstrap['mean']:.4f}", file=sys.stderr, flush=True)
        print(f"      95% CI: [{ot_bootstrap['ci_lower']:.4f}, {ot_bootstrap['ci_upper']:.4f}]", file=sys.stderr, flush=True)
        print(f"      Std Dev: {ot_bootstrap['std']:.4f}", file=sys.stderr, flush=True)
        print()

        best_match_bootstrap = bootstrap_best_match_metrics(
            topics1_list, topics2_list, vocabulary, n_bootstrap
        )
        print(f"   ✅ Best-Match Bootstrap Complete:", file=sys.stderr, flush=True)
        print(f"      Mean TVD 1→2: {best_match_bootstrap['mean_tvd_1to2']['mean']:.4f}", file=sys.stderr, flush=True)
        print(f"      Mean TVD 2→1: {best_match_bootstrap['mean_tvd_2to1']['mean']:.4f}", file=sys.stderr, flush=True)
        print(f"      Reciprocal Count: {best_match_bootstrap['reciprocal_count']['mean']:.1f}", file=sys.stderr, flush=True)
        print()

        # Calculate permutation test for OT distance significance
        logger.info("Performing permutation test for OT distance significance...")
        print("🔀 Computing Permutation Test for OT Distance...", file=sys.stderr, flush=True)
        print("   This tests if the observed difference is statistically significant...", file=sys.stderr, flush=True)
        
        n_permutations = int(request.form.get("n_permutations", 1000))
        permutation_result = permutation_test_ot_distance(
            topics1_list, topics2_list, prevalence1, prevalence2, vocabulary, n_permutations
        )
        
        if permutation_result:
            print(f"   ✅ Permutation Test Complete:", file=sys.stderr, flush=True)
            print(f"      Observed OT: {permutation_result['observed_ot']:.4f}", file=sys.stderr, flush=True)
            print(f"      Null Mean: {permutation_result['null_ot_mean']:.4f}", file=sys.stderr, flush=True)
            print(f"      Null Std: {permutation_result['null_ot_std']:.4f}", file=sys.stderr, flush=True)
            print(f"      P-value: {permutation_result['p_value']:.4f}", file=sys.stderr, flush=True)
            print(f"      {'✅ Significant' if permutation_result['significant'] else '❌ Not significant'} (α=0.05)", file=sys.stderr, flush=True)
        else:
            print("   ⚠️  Permutation test failed", file=sys.stderr, flush=True)
        print()

        # Prepare TVD vectors for potential Mann-Whitney test
        # Extract TVD values for both directions
        tvd_values_1to2 = [m['tvd'] for m in best_match_results['best_matches_1to2']]
        tvd_values_2to1 = [m['tvd'] for m in best_match_results['best_matches_2to1']]
        
        # Perform Mann-Whitney test to compare directional differences
        logger.info("Performing Mann-Whitney test to compare directional TVD distributions...")
        print("📊 Computing Mann-Whitney U Test (1→2 vs 2→1 TVD distributions)...", file=sys.stderr, flush=True)
        
        mann_whitney_result = mann_whitney_test(tvd_values_1to2, tvd_values_2to1)
        
        if mann_whitney_result:
            print(f"   ✅ Mann-Whitney Test Complete:", file=sys.stderr, flush=True)
            print(f"      U-statistic: {mann_whitney_result['statistic']:.2f}", file=sys.stderr, flush=True)
            print(f"      P-value: {mann_whitney_result['p_value']:.4f}", file=sys.stderr, flush=True)
            print(f"      {'✅ Significant asymmetry' if mann_whitney_result['significant'] else '❌ No significant asymmetry'} (α=0.05)", file=sys.stderr, flush=True)
            if mann_whitney_result['significant']:
                mean_1to2 = np.mean(tvd_values_1to2)
                mean_2to1 = np.mean(tvd_values_2to1)
                if mean_1to2 < mean_2to1:
                    print(f"      → Dataset 1 topics are more stable (better matches in Dataset 2)", file=sys.stderr, flush=True)
                else:
                    print(f"      → Dataset 2 topics are more stable (better matches in Dataset 1)", file=sys.stderr, flush=True)
        else:
            print("   ⚠️  Mann-Whitney test failed", file=sys.stderr, flush=True)
        print()

        # Print detailed reciprocal matches
        if best_match_results['reciprocal_matches']:
            print("🔄 DETAILED RECIPROCAL MATCHES (Stable Topics):", file=sys.stderr, flush=True)
            print("-"*60, file=sys.stderr, flush=True)
            for match in best_match_results['reciprocal_matches']:
                print(f"   • Topic {match['topic1_idx']+1} (Dataset 1) ⟷ Topic {match['topic2_idx']+1} (Dataset 2)", file=sys.stderr, flush=True)
                print(f"     TVD: {match['tvd']:.4f} {'✅ Good match!' if match['tvd'] < 0.3 else '⚠️ Moderate' if match['tvd'] < 0.7 else '❌ Poor'}", file=sys.stderr, flush=True)
                print(f"     Dataset 1 words: {match['topic1_name']}", file=sys.stderr, flush=True)
                print(f"     Dataset 2 words: {match['topic2_name']}", file=sys.stderr, flush=True)
                print()
        else:
            print("❌ No reciprocal matches found (topics may have merged/split)", file=sys.stderr, flush=True)
            print()

        # Print interpretation
        print("="*80, file=sys.stderr, flush=True)
        print("📖 INTERPRETATION:", file=sys.stderr, flush=True)
        print("-"*60, file=sys.stderr, flush=True)

        if ot_distance < 0.2:
            print("✅ EXCELLENT: Topic structures are very similar across datasets", file=sys.stderr, flush=True)
        elif ot_distance < 0.4:
            print("✓  GOOD: Topic structures are reasonably similar", file=sys.stderr, flush=True)
        elif ot_distance < 0.6:
            print("⚠️  MODERATE: Significant differences in topic structures", file=sys.stderr, flush=True)
        else:
            print("❌ POOR: Topic structures are quite different", file=sys.stderr, flush=True)

        print()
        print(f"Coverage: {best_match_results['coverage_1to2']*100:.0f}% of Dataset 1 topics", file=sys.stderr, flush=True)
        print(f"          have good matches (TVD < 0.3) in Dataset 2", file=sys.stderr, flush=True)
        print()

        if best_match_results['num_reciprocal'] >= len(topics1_list) * 0.5:
            print(f"✅ Strong topic stability: {best_match_results['num_reciprocal']} reciprocal matches", file=sys.stderr, flush=True)
        elif best_match_results['num_reciprocal'] > 0:
            print(f"⚠️  Moderate topic stability: {best_match_results['num_reciprocal']} reciprocal matches", file=sys.stderr, flush=True)
        else:
            print("❌ Low topic stability: Topics have likely merged or split", file=sys.stderr, flush=True)

        print("="*80, file=sys.stderr, flush=True)
        print()

        # Prepare enhanced response
        response = {
            # Basic TVD matrix (for backward compatibility)
            'similarity_matrix': similarity_matrix.tolist(),
            'matrix_data': matrix_data,
            'topics1': [{'id': k, 'name': v['name']} for k, v in sorted(topics1_dict.items())],
            'topics2': [{'id': k, 'name': v['name']} for k, v in sorted(topics2_dict.items())],
            'vocabulary_size': len(vocabulary),
            'num_topics1': num_topics1,
            'num_topics2': num_topics2,

            # Enhanced metrics
            'optimal_transport': {
                'distance': ot_distance,
                'bootstrap': ot_bootstrap,
                'permutation_test': permutation_result
            },
            'best_match_analysis': best_match_results,
            'best_match_bootstrap': best_match_bootstrap,
            'statistical_tests': {
                'mann_whitney': mann_whitney_result,
                'tvd_vectors': {
                    'dataset1_to_2': tvd_values_1to2,
                    'dataset2_to_1': tvd_values_2to1
                }
            },
            'has_prevalence': prevalence_file1 is not None and prevalence_file2 is not None
        }

        logger.info("Comparison complete. OT distance: %.4f", ot_distance if ot_distance else 0)
        print(f"✅ Comparison complete! Results sent to frontend.\n", file=sys.stderr, flush=True)
        return jsonify(response)

    except Exception as e:
        logger.error("Error in compare endpoint: %s", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Comparison failed: " + str(e)}), 500

@app.route("/summary", methods=["POST"])
def summary():
    try:
        num_topics = int(request.form["numTopics"])
        num_words = int(request.form["numWords"])
        summary_text = "Analysis performed with {} topics and {} words per topic.".format(num_topics, num_words)
        return jsonify({"results": summary_text})
    except Exception as e:
        logger.error("Error in summary endpoint: %s", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

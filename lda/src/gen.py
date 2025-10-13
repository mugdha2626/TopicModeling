# -*- coding: utf-8 -*-
import os
import zipfile
import shutil
import tempfile
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader
from gensim import corpora
from gensim.models import HdpModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lemmatizer = None
logger.info("Using built-in text preprocessing (NLTK downloads skipped)")

try:
    from nltk.corpus import wordnet
    wordnet.synsets("test") 
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK WordNet found and loaded")
except:
    logger.info("NLTK not available, using basic preprocessing without lemmatization")

def extract_year_from_title(file_name):
    """Extract year from filename using regex."""
    match = re.search(r"- (\d{4}) -", file_name)
    return int(match.group(1)) if match else None

def clean_pdf_text(text):
    """
    Advanced text preprocessing for better HDP results:
    - Remove special characters, numbers, URLs, emails
    - Fix common OCR errors and PDF artifacts
    - Convert to lowercase and filter short words
    - Lemmatize words and remove stopwords
    """
    if not text or len(text) < 100:
        return ""
    
    # Remove URLs, emails, and special patterns
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)  # Remove standalone numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    
    # Fix common PDF extraction artifacts
    text = re.sub(r'\b\w{1,2}\b', ' ', text)     # Remove 1-2 letter words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)     # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize whitespace
    text = text.lower()
    
    # Enhanced stopwords for academic papers with fallback
    try:
        english_stopwords = set(stopwords.words("english"))
    except:
        # Fallback English stopwords if NLTK download failed
        english_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', 'should', 'now'
        }
    
    academic_stopwords = {
        'study', 'research', 'studies', 'analysis', 'results', 'method', 'methods',
        'data', 'table', 'figure', 'findings', 'conclusion', 'abstract', 'introduction',
        'discussion', 'participants', 'participant', 'experiment', 'experiments',
        'university', 'doi', 'journal', 'published', 'authors', 'author',
        'paper', 'papers', 'article', 'articles', 'shows', 'shown', 'show',
        'using', 'used', 'use', 'based', 'may', 'also', 'however', 'therefore',
        'thus', 'furthermore', 'moreover', 'additionally', 'finally', 'first',
        'second', 'third', 'last', 'next', 'previous', 'following', 'above',
        'below', 'within', 'across', 'between', 'among', 'through', 'during',
        'before', 'after', 'since', 'until', 'while', 'although', 'though',
        'whereas', 'unless', 'whether', 'either', 'neither', 'both', 'all',
        'each', 'every', 'any', 'some', 'many', 'few', 'much', 'little',
        'more', 'most', 'less', 'least', 'only', 'just', 'even', 'still',
        'yet', 'already', 'again', 'once', 'twice', 'always', 'never',
        'often', 'sometimes', 'usually', 'generally', 'particularly',
        'especially', 'specifically', 'exactly', 'precisely', 'approximately',
        'roughly', 'about', 'around', 'nearly', 'almost', 'quite', 'rather',
        'very', 'extremely', 'highly', 'completely', 'entirely', 'fully',
        'partially', 'partly', 'mainly', 'mostly', 'largely'
    }
    all_stopwords = english_stopwords.union(academic_stopwords)
    
    # Tokenize and filter
    tokens = text.split()
    filtered_tokens = []
    
    for token in tokens:
        # Keep only words that are 3+ characters, alphabetic, and not stopwords
        if (len(token) >= 3 and 
            token.isalpha() and 
            token not in all_stopwords):
            
            # Apply lemmatization if available
            if lemmatizer:
                token = lemmatizer.lemmatize(token)
            
            filtered_tokens.append(token)
    
    processed_text = " ".join(filtered_tokens)
    
    # Return empty if text is too short after preprocessing
    if len(processed_text) < 50:
        return ""
        
    return processed_text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logger.error("Error processing %s: %s", pdf_path, str(e))
    return text

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

def evaluate_hdp_model(hdp_model, corpus, dictionary, doc_texts):
    """Comprehensive evaluation of HDP model quality."""
    print("\n" + "="*80)
    print("*** HDP MODEL ANALYSIS RESULTS ***")
    print("="*80)
    
    # Basic model stats - filter out very weak topics (much more aggressive)
    all_topics = hdp_model.get_topics()
    significant_topic_indices = [i for i, t in enumerate(all_topics) if np.sum(t) > 0.5]  # Much higher threshold
    num_topics = len(significant_topic_indices)
    print("Number of active topics: {}".format(num_topics))
    print("Number of documents: {}".format(len(corpus)))
    print("Vocabulary size: {}".format(len(dictionary)))
    
    # Model parameters (HDP uses different attributes)
    print("Alpha (doc-topic): {}".format(getattr(hdp_model, 'm_alpha', 'N/A')))
    print("Gamma (topic concentration): {}".format(getattr(hdp_model, 'm_gamma', 'N/A')))
    
    # Document-topic distributions
    doc_topics = []
    for doc_bow in corpus:
        doc_topic_dist = hdp_model[doc_bow]
        doc_topics.append(doc_topic_dist)
    
    # Topic coherence analysis
    print("\n*** DETAILED TOPIC BREAKDOWN:***")
    print("-"*60)
    
    topic_info = []
    for i, topic_idx in enumerate(significant_topic_indices):
        topic_words = hdp_model.show_topic(topic_idx, topn=10)
        top_words = [word for word, prob in topic_words]
        probabilities = [prob for word, prob in topic_words]
        
        # Count documents strongly associated with this topic (>20% probability)
        strong_docs = 0
        topic_strengths = []
        
        for doc_topic_dist in doc_topics:
            topic_prob = dict(doc_topic_dist).get(topic_idx, 0)
            topic_strengths.append(topic_prob)
            if topic_prob > 0.2:
                strong_docs += 1
        
        avg_strength = np.mean(topic_strengths) if topic_strengths else 0
        max_strength = np.max(topic_strengths) if topic_strengths else 0
        
        print("\n*** TOPIC {}: ***".format(i + 1))
        print("   Top words: {}".format(', '.join(top_words)))
        print("   Word probabilities: {}".format(["{:.4f}".format(p) for p in probabilities[:5]]))
        print("   Documents with >20% loading: {}/{} ({:.1f}%)".format(strong_docs, len(corpus), strong_docs/len(corpus)*100))
        print("   Average topic strength: {:.3f}".format(avg_strength))
        print("   Maximum topic strength: {:.3f}".format(max_strength))
        
        topic_info.append({
            'topic_id': i + 1,
            'top_words': top_words,
            'probabilities': probabilities,
            'strong_docs': strong_docs,
            'avg_strength': avg_strength,
            'max_strength': max_strength
        })
        
        print("-"*60)
    
    # Overall model quality assessment
    print("\n*** MODEL QUALITY ASSESSMENT: ***")
    print("-"*60)
    
    # Calculate topic concentration
    doc_topic_concentrations = []
    for doc_topic_dist in doc_topics:
        if doc_topic_dist:
            max_topic_prob = max([prob for _, prob in doc_topic_dist])
            doc_topic_concentrations.append(max_topic_prob)
    
    avg_concentration = np.mean(doc_topic_concentrations) if doc_topic_concentrations else 0
    print("Average document-topic concentration: {:.3f}".format(avg_concentration))
    
    # Topic separation quality
    if avg_concentration > 0.3:
        separation_quality = "Good"
    elif avg_concentration > 0.2:
        separation_quality = "Fair"
    else:
        separation_quality = "Poor"
    
    print("Topic separation quality: {}".format(separation_quality))
    
    # Recommendations
    print("\n*** RECOMMENDATIONS: ***")
    if num_topics < 3:
        print("   WARNING: Very few topics detected:")
        print("      - Your corpus might be too small or homogeneous")
        print("      - Consider lowering gamma parameter")
        print("      - Check if preprocessing is too aggressive")
    
    if avg_concentration < 0.2:
        print("   WARNING: Low topic concentration suggests:")
        print("      - Topics may be too generic or overlapping")
        print("      - Consider increasing alpha parameter")
        print("      - Check document preprocessing quality")
    
    if num_topics > 20:
        print("   WARNING: Many topics detected:")
        print("      - Some topics might be very specific")
        print("      - Consider filtering topics with low document support")
        print("      - Increase gamma to encourage fewer topics")
    
    print("="*80)
    return topic_info

def create_topic_visualizations(hdp_model, dictionary, topic_info, num_words=10):
    """Create comprehensive topic visualizations."""
    num_topics = len(topic_info)
    
    if num_topics == 0:
        print("No topics to visualize!")
        return
    
    # 1. Bar chart visualization
    if num_topics <= 8:
        num_cols = min(4, num_topics)
        num_rows = (num_topics + num_cols - 1) // num_cols
        fig_size = (16, 4 * num_rows)
    else:
        num_cols = 4
        num_rows = (num_topics + num_cols - 1) // num_cols
        fig_size = (16, 3 * num_rows)

    plt.figure(figsize=fig_size)
    plt.suptitle("HDP Topic Analysis - Top Words per Topic", fontsize=16, y=0.98)
    
    for i, topic in enumerate(topic_info):
        plt.subplot(num_rows, num_cols, i + 1)
        words = topic['top_words'][:num_words]
        probs = topic['probabilities'][:num_words]
        
        bars = plt.barh(range(len(words)), probs, color=plt.cm.Set3(i))
        plt.yticks(range(len(words)), words)
        plt.xlabel('Probability')
        plt.title("Topic {}\n({} docs)".format(topic['topic_id'], topic['strong_docs']), fontsize=10)
        plt.gca().invert_yaxis()
        
        # Add probability labels on bars
        for j, (bar, prob) in enumerate(zip(bars, probs)):
            plt.text(prob + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Topic strength heatmap
    if num_topics > 1:
        plt.figure(figsize=(12, 6))
        
        # Create data for heatmap
        topic_ids = [t['topic_id'] for t in topic_info]
        metrics = ['Avg Strength', 'Max Strength', 'Doc Coverage']
        
        heatmap_data = []
        for topic in topic_info:
            row = [
                topic['avg_strength'],
                topic['max_strength'],
                topic['strong_docs'] / len(topic_info)  # Normalized doc coverage
            ]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data).T
        
        sns.heatmap(heatmap_data, 
                   xticklabels=["Topic {}".format(tid) for tid in topic_ids],
                   yticklabels=metrics,
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Strength/Coverage'})
        
        plt.title("Topic Quality Metrics Heatmap")
        plt.xlabel("Topics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 3. Word clouds for top topics
    top_3_topics = sorted(topic_info, key=lambda x: x['avg_strength'], reverse=True)[:3]
    
    if len(top_3_topics) > 0:
        fig, axes = plt.subplots(1, len(top_3_topics), figsize=(5*len(top_3_topics), 5))
        if len(top_3_topics) == 1:
            axes = [axes]
        
        fig.suptitle("Word Clouds for Top Topics", fontsize=16)
        
        for i, topic in enumerate(top_3_topics):
            word_freq = dict(zip(topic['top_words'][:20], topic['probabilities'][:20]))
            
            try:
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    max_words=20,
                                    colormap='viridis').generate_from_frequencies(word_freq)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title("Topic {}".format(topic['topic_id']), fontsize=12)
                axes[i].axis('off')
            except Exception as e:
                logger.warning("Could not create word cloud for topic %s: %s", topic['topic_id'], str(e))
                axes[i].text(0.5, 0.5, "Topic {}\nWord cloud unavailable".format(topic['topic_id']), 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def export_topic_results(topic_info, hdp_model, output_file="hdp_topics_analysis.csv"):
    """Export topic analysis results to CSV."""
    export_data = []
    
    for topic in topic_info:
        for i, (word, prob) in enumerate(zip(topic['top_words'], topic['probabilities'])):
            export_data.append({
                'Topic_ID': topic['topic_id'],
                'Word_Rank': i + 1,
                'Word': word,
                'Probability': prob,
                'Topic_Avg_Strength': topic['avg_strength'],
                'Topic_Max_Strength': topic['max_strength'],
                'Strong_Documents': topic['strong_docs']
            })
    
    df = pd.DataFrame(export_data)
    df.to_csv(output_file, index=False)
    logger.info("Topic analysis exported to %s", output_file)
    return df

### Main Pipeline ###
def main(zip_path=None):
    """
    Main HDP analysis pipeline.
    
    Args:
        zip_path: Path to ZIP file containing PDFs. If None, uses default path.
    """
    if zip_path is None:
        # Default path - update this to your ZIP file location
        zip_path = "/Users/manas/Desktop/research/LDATopicModeling/lda/pdfs.zip"
    
    logger.info("Starting HDP analysis on: %s", zip_path)
    
    # Check if file exists
    if not os.path.exists(zip_path):
        logger.error("ZIP file not found: %s", zip_path)
        print("ERROR: ZIP file not found at {}".format(zip_path))
        print("Please update the zip_path in the main() function or pass it as an argument.")
        return
    
    # Create temporary directory for extraction
    extracted_path = tempfile.mkdtemp()
    logger.info("Extracting files to: %s", extracted_path)
    
    try:
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
        
        # Process PDFs
        pdf_texts = []
        titles = []
        processed_files = 0
        
        logger.info("Processing PDF files...")
        for root, _, files in os.walk(extracted_path):
            for file_name in files:
                if file_name.endswith(".pdf"):
                    pdf_path = os.path.join(root, file_name)
                    logger.debug("Processing: %s", file_name)
                    
                    # Extract and clean text
                    raw_text = extract_text_from_pdf(pdf_path)
                    clean_text = clean_pdf_text(raw_text)
                    
                    if clean_text:  # Only add if text extraction successful
                        pdf_texts.append(clean_text)
                        titles.append(file_name)
                        processed_files += 1
                        logger.debug("Successfully processed: %s", file_name)
                    else:
                        logger.warning("WARNING: Skipped (no clean text): %s", file_name)
        
        logger.info("Successfully processed %d PDF files", processed_files)
        
        if processed_files == 0:
            logger.error("No PDF files were successfully processed!")
            return
        
        print("\nProcessing complete: {} documents loaded".format(processed_files))
        
        # Create document-term matrix with improved parameters
        logger.info("Creating document-term matrix...")
        vectorizer = CountVectorizer(
            max_df=0.7,           # More restrictive - exclude words in >70% of documents
            min_df=max(2, processed_files // 10),  # Higher minimum frequency
            max_features=200,     # Much smaller vocabulary for small corpus
            token_pattern=r'\b[a-zA-Z]{4,}\b'  # Only 4+ letter words (more restrictive)
        )
        
        doc_term_matrix = vectorizer.fit_transform(pdf_texts)
        vocab = vectorizer.get_feature_names_out()
        
        logger.info("Created document-term matrix: %s", str(doc_term_matrix.shape))
        logger.info("Vocabulary size: %d", len(vocab))
        
        # Prepare Gensim corpus and dictionary
        logger.info("Preparing Gensim corpus...")
        corpus, dictionary = prepare_gensim_corpus(doc_term_matrix, vectorizer)
        
        # Initialize and train HDP model with much more restrictive parameters
        logger.info("Training HDP model...")
        hdp_model = HdpModel(
            corpus=corpus, 
            id2word=dictionary, 
            random_state=42,
            alpha=0.1,           # Document-topic concentration (much lower)
            gamma=0.01,          # Topic concentration parameter (very low = very few topics)
            eta=0.5,             # Topic-word concentration (higher = more focused topics)
            T=10,                # Top level truncation (much smaller)
            K=5,                 # Second level truncation (much smaller)
            kappa=0.5,           # Learning rate
            tau=1.0              # Learning parameter
        )
        
        logger.info("HDP model training complete!")
        
        # Comprehensive model evaluation
        topic_info = evaluate_hdp_model(hdp_model, corpus, dictionary, pdf_texts)
        
        if not topic_info:
            logger.warning("No topics were discovered!")
            return
        
        # Create visualizations
        logger.info("Creating visualizations...")
        create_topic_visualizations(hdp_model, dictionary, topic_info, num_words=10)
        
        # Export results
        logger.info("Exporting results...")
        results_df = export_topic_results(topic_info, hdp_model, 
                                        output_file="hdp_analysis_{}_docs.csv".format(processed_files))
        
        print("\nAnalysis complete!")
        print("Discovered {} topics from {} documents".format(len(topic_info), processed_files))
        print("Results exported to: hdp_analysis_{}_docs.csv".format(processed_files))
        
        return {
            'hdp_model': hdp_model,
            'topic_info': topic_info,
            'corpus': corpus,
            'dictionary': dictionary,
            'results_df': results_df
        }
        
    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        raise
    finally:
        # Clean up temporary directory
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(extracted_path)


if __name__ == "__main__":
    # You can specify a custom ZIP path here or modify the default in main()
    import sys
    
    if len(sys.argv) > 1:
        # Run with custom ZIP path: python gen.py /path/to/your/file.zip
        custom_zip_path = sys.argv[1]
        main(custom_zip_path)
    else:
        main()


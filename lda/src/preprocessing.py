"""
Shared preprocessing module for topic modeling.
Used by both app.py (Flask backend) and gen.py (standalone script).
"""

import re
import logging
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download("stopwords", quiet=True)
    _test_stops = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    _test_stem = stemmer.stem("studies")
    logger.info("NLTK properly initialized with %d stopwords", len(_test_stops))
except Exception as e:
    logger.error("NLTK initialization failed: %s. Will use fallback.", str(e))
    stemmer = None


# Enhanced stopwords for academic/research papers
ENHANCED_STOPWORDS = {
    # Academic/research structural words
    'study', 'research', 'studies', 'analysis', 'results', 'method', 'methods',
    'table', 'figure', 'fig', 'findings', 'conclusion', 'abstract', 'introduction',
    'discussion', 'participants', 'participant', 'experiment', 'experiments',
    'university', 'doi', 'journal', 'published', 'authors', 'author', 'using', 'used',
    'show', 'shown', 'shows', 'may', 'also', 'however', 'therefore', 'thus',
    'furthermore', 'moreover', 'additionally', 'meanwhile', 'nonetheless',
    'supplementary', 'appendix', 'volume', 'article', 'paper', 'review',
    'respectively', 'whereas', 'although', 'whether', 'overall',

    # Common academic verbs (too generic to be topical)
    'suggest', 'suggests', 'suggested', 'indicate', 'indicates', 'indicated',
    'reveal', 'reveals', 'revealed', 'demonstrate', 'demonstrates', 'demonstrated',
    'provide', 'provides', 'provided', 'involve', 'involves', 'involved',
    'include', 'includes', 'included', 'identify', 'identifies', 'identified',
    'report', 'reports', 'reported', 'propose', 'proposes', 'proposed',
    'require', 'requires', 'required', 'consider', 'considers', 'considered',
    'enable', 'enables', 'enabled',

    # Common academic adjectives/adverbs (too generic)
    'significant', 'significantly', 'statistical', 'statistically',
    'previous', 'previously', 'current', 'currently', 'recent', 'recently',
    'possible', 'potential', 'potentially', 'various', 'several', 'likely',
    'similar', 'consistent', 'relevant', 'important', 'importantly',
    'higher', 'lower', 'greater', 'less', 'fewer',
    'observe', 'observed', 'difference', 'differences', 'finding', 'findings',
    'examine', 'examined', 'describe', 'described', 'determine', 'determined',
    'assess', 'assessed', 'evaluate', 'evaluated', 'investigate', 'investigated',
    'explore', 'explored', 'confirm', 'confirmed', 'establish', 'established',
    'conduct', 'conducted', 'publish', 'perform', 'obtain', 'select', 'selected',
    'apply', 'applied', 'test', 'tested', 'note', 'noted', 'aim', 'aimed',
    'find', 'found', 'compare', 'increase', 'decrease', 'relate',
    'correspond', 'correspond', 'support', 'supported', 'predict', 'predicted',

    # Generic quantitative/study terms
    'number', 'value', 'values', 'result', 'different', 'time', 'times',
    'based', 'two', 'one', 'three', 'four', 'five', 'first',
    'second', 'third', 'can', 'use', 'within', 'across', 'between', 'among',
    'well', 'large', 'small', 'high', 'low', 'new', 'different', 'same',
    'specific', 'general', 'particular', 'example', 'examples', 'case', 'cases',
    'observed', 'obtained', 'performed', 'present',
    'related', 'associated', 'compared', 'due', 'examined', 'found',
    'increased', 'decreased', 'range', 'level', 'levels', 'term', 'terms',
    'group', 'groups', 'sample', 'samples', 'control', 'controls',
    'region', 'regions', 'area', 'areas',
    'individual', 'individuals', 'subject', 'subjects', 'patient', 'patients',
    'mean', 'average', 'total', 'score', 'scores', 'percentage',
    'manuscript', 'material', 'materials', 'see',
    'base', 'back', 'like', 'given', 'take', 'taken',
    'age', 'aged', 'male', 'female', 'sex', 'gender', 'year', 'years',
    'model', 'models', 'response', 'responses',
    'task', 'tasks', 'trial', 'trials', 'block', 'blocks',
    'correlation', 'correlations', 'anova', 'regression', 'ttest',
    'paired', 'baseline', 'standard', 'deviation', 'error', 'variance',
    'image', 'images', 'contrast',

    # Additional generic terms
    'condition', 'conditions', 'reason', 'reasons',
    'together', 'factor', 'factors', 'refer', 'refers',
    'approach', 'approaches', 'technique', 'techniques', 'problem', 'problems',
    'solution', 'solutions', 'measure', 'measures', 'measured',
    'assumption', 'assumptions', 'plan', 'plans', 'speak', 'speaking',
    'qwen', 'effect', 'effects', 'change', 'changes',

    # Words that are TOO generic even in CS
    'system', 'systems', 'process', 'method', 'methods', 'data',

    # Common place names (from author affiliations)
    'germany', 'london', 'york', 'california', 'missouri', 'china', 'italy',
    'france', 'england', 'boston', 'chicago', 'amsterdam', 'oxford', 'cambridge',
    'stanford', 'harvard', 'princeton', 'berkeley', 'toronto', 'montreal',
    'manchester', 'edinburgh', 'glasgow', 'dublin', 'paris', 'berlin', 'munich',
    'tokyo', 'korea', 'japan', 'australia', 'sydney', 'melbourne', 'canada',

    # Common surnames (from author names leaking into text)
    'smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis',
    'wilson', 'moore', 'taylor', 'anderson', 'thomas', 'jackson', 'white',
    'harris', 'martin', 'thompson', 'garcia', 'martinez', 'robinson',
    'clark', 'rodriguez', 'lewis', 'lee', 'walker', 'hall', 'allen',
    'young', 'hernandez', 'king', 'wright', 'lopez', 'hill', 'scott',
    'green', 'adams', 'baker', 'gonzalez', 'nelson', 'carter', 'mitchell',
    'perez', 'roberts', 'turner', 'phillips', 'campbell', 'parker', 'evans',
    'edwards', 'collins', 'stewart', 'sanchez', 'morris', 'rogers', 'reed',
    'cook', 'morgan', 'bell', 'murphy', 'bailey', 'rivera', 'cooper',
    'richardson', 'cox', 'howard', 'ward', 'torres', 'peterson', 'gray',
    'ramirez', 'james', 'watson', 'brooks', 'kelly', 'sanders', 'price',
    'bennett', 'wood', 'barnes', 'ross', 'henderson', 'coleman', 'jenkins',
    'perry', 'powell', 'long', 'patterson', 'hughes', 'flores', 'washington',
    'butler', 'simmons', 'foster', 'gonzales', 'bryant', 'alexander', 'russell',
    'griffin', 'diaz', 'hayes', 'myers', 'ford', 'hamilton', 'graham', 'sullivan',
    'wallace', 'woods', 'cole', 'west', 'jordan', 'owens', 'reynolds', 'fisher',
    'ellis', 'harrison', 'gibson', 'mcdonald', 'cruz', 'marshall', 'ortiz',
    'gomez', 'murray', 'freeman', 'wells', 'webb', 'simpson', 'stevens',
    'tucker', 'porter', 'hunter', 'hicks', 'crawford', 'henry', 'boyd',
    'mason', 'morales', 'kennedy', 'warren', 'dixon', 'ramos', 'reyes',
    'burns', 'gordon', 'shaw', 'holmes', 'rice', 'robertson', 'hunt',
    'black', 'daniels', 'palmer', 'mills', 'nichols', 'grant', 'knight',
    'ferguson', 'rose', 'stone', 'hawkins', 'dunn', 'perkins', 'hudson',
    'spencer', 'gardner', 'stephens', 'payne', 'pierce', 'berry', 'matthews',
    'arnold', 'wagner', 'willis', 'ray', 'watkins', 'olson', 'carroll',
    'duncan', 'snyder', 'hart', 'cunningham', 'bradley', 'lane', 'andrews',
    'ruiz', 'harper', 'fox', 'riley', 'armstrong', 'carpenter', 'weaver',
    'greene', 'lawrence', 'elliott', 'chavez', 'sims', 'austin', 'peters',
    'kelley', 'franklin', 'lawson', 'fields', 'gutierrez', 'ryan', 'schmidt',
    'carr', 'vasquez', 'castillo', 'wheeler', 'chapman', 'oliver', 'montgomery',
    'richards', 'williamson', 'johnston', 'banks', 'meyer', 'bishop', 'mccoy',
    'howell', 'alvarez', 'morrison', 'hansen', 'fernandez', 'garza', 'harvey',
    'little', 'burton', 'stanley', 'nguyen', 'george', 'jacobs', 'reid',
    'kim', 'fuller', 'lynch', 'dean', 'gilbert', 'garrett', 'romero',
    'welch', 'larson', 'frazier', 'burke', 'hanson', 'day', 'mendoza',
    'moreno', 'bowman', 'medina', 'fowler', 'brewer', 'hoffman', 'carlson',
    'silva', 'pearson', 'holland', 'douglas', 'fleming', 'jensen', 'vargas',
    'byrd', 'davidson', 'hopkins', 'may', 'terry', 'herrera', 'wade',
    'soto', 'walters', 'curtis', 'neal', 'caldwell', 'lowe', 'jennings',
    'barnett', 'graves', 'jimenez', 'horton', 'shelton', 'barrett', 'obrien',
    'castro', 'sutton', 'gregory', 'mckinney', 'lucas', 'miles', 'craig',
    'rodriquez', 'chambers', 'holt', 'lambert', 'fletcher', 'watts', 'bates',
    'hale', 'rhodes', 'pena', 'beck', 'newman', 'haynes', 'mcdaniel',
    'mendez', 'bush', 'vaughn', 'parks', 'dawson', 'santiago', 'norris',
    'hardy', 'love', 'steele', 'curry', 'powers', 'schultz', 'barker',
    'guzman', 'page', 'munoz', 'ball', 'keller', 'chandler', 'weber',
    'leonard', 'walsh', 'lyons', 'ramsey', 'wolfe', 'schneider', 'mullins',
    'benson', 'sharp', 'bowen', 'daniel', 'barber', 'cummings', 'hines',
    'baldwin', 'griffith', 'valdez', 'hubbard', 'salazar', 'reeves', 'warner',
    'stevenson', 'burgess', 'santos', 'tate', 'cross', 'garner', 'mann',
    'mack', 'moss', 'thornton', 'dennis', 'mcgee', 'farmer', 'delgado',
    'aguilar', 'vega', 'glover', 'manning', 'cohen', 'harmon', 'rodgers',
    'robbins', 'newton', 'todd', 'blair', 'higgins', 'ingram', 'reese',
    'cannon', 'strickland', 'townsend', 'potter', 'goodwin', 'walton',
    'rowe', 'hampton', 'ortega', 'patton', 'swanson', 'joseph', 'francis',
    'goodman', 'maldonado', 'yates', 'becker', 'erickson', 'hodges',
    'rios', 'conner', 'adkins', 'webster', 'norman', 'malone', 'hammond',
    'flowers', 'cobb', 'moody', 'quinn', 'blake', 'maxwell', 'pope',
    'floyd', 'osborne', 'paul', 'mccarthy', 'guerrero', 'lindsey', 'estrada',
    'sandoval', 'gibbs', 'tyler', 'gross', 'fitzgerald', 'stokes', 'doyle',
    'sherman', 'saunders', 'wise', 'colon', 'gill', 'alvarado', 'greer',
    'padilla', 'simon', 'waters', 'nunez', 'ballard', 'schwartz', 'mcbride',

    # Academic metadata terms
    'pmcid', 'pubmed', 'doi', 'issn', 'isbn', 'copyright', 'license',
    'elsevier', 'springer', 'wiley', 'pergamon', 'press', 'publisher', 'publication',
    'nih', 'medline', 'crossref', 'pubmedcentral', 'manuscript', 'available',
    'open', 'access', 'online', 'print', 'epub', 'accepted', 'received', 'revised',
    'word', 'words', 'item', 'items', 'update',

    # Experimental methodology words (describe HOW, not WHAT research is about)
    'letter', 'letters', 'sentence', 'sentences', 'paragraph',
    'stimulus', 'stimuli', 'target', 'targets', 'probe', 'probes',
    'cue', 'cues', 'trial', 'trials', 'block', 'blocks',
    'load', 'loading', 'face', 'faces', 'object', 'objects',
    'location', 'locations', 'display', 'scene', 'scenes',
    'left', 'right', 'hand', 'hands', 'button', 'press',
    'adult', 'adults', 'child', 'children', 'infant', 'infants',
    'adolescent', 'adolescents', 'participant', 'participants',
    'period', 'rest', 'resting', 'session', 'sessions',
    'accuracy', 'correct', 'incorrect', 'speed', 'slow', 'fast',
    'train', 'training', 'trained', 'learn', 'learned', 'learning',
    'connectivity', 'connected', 'connection', 'connections',
    'delay', 'delayed', 'encode', 'encoding', 'encoded',
    'maintenance', 'maintain', 'maintained',
    'switch', 'switching', 'refresh', 'interference',
    'post', 'dual', 'skill', 'skills', 'rule', 'rules',
    'outcome', 'outcomes', 'recognition', 'meta', 'activation',
    'signal', 'signals', 'inhibition', 'frontal', 'temporal',
    'prefrontal', 'network', 'networks', 'deficit', 'deficits',
    'disorder', 'disorders', 'negative', 'positive',
    'list', 'lists', 'semantic', 'developmental', 'sensory', 'perceptual',
    'quarterly', 'library', 'comply', 'null', 'code', 'mail', 'free',
    'half', 'edge', 'unite', 'drop', 'miss', 'born', 'risk',
    'easy', 'hard', 'quickly', 'slowly', 'fully',
    'month', 'months', 'week', 'weeks', 'day', 'days',
    'transfer', 'intervention', 'improvement', 'eye', 'eyes',
    'array', 'picture', 'pictures', 'category', 'categories',
    'parietal', 'occipital',

    # Working memory researcher surnames (leak from citations)
    'engle', 'klingberg', 'miyake', 'oberauer', 'gathercole', 'daneman',
    'caplan', 'alloway', 'barrouillet', 'vogel', 'egner', 'kiyonaga',
    'luria', 'baddeley', 'hitch', 'cowan', 'luck', 'awh', 'jonides',
    'kane', 'conway', 'unsworth', 'redick', 'shipstead', 'jaeggi',
    'logie', 'repovs', 'chein', 'postle', 'salthouse', 'verhaeghen',
    'mammarella', 'koziol', 'slotnick', 'balaban', 'lorenz', 'reuter',
    'teng', 'peng', 'buschkuehl', 'crone', 'courtney', 'rypma',
    'friedman', 'hambrick', 'hasher', 'lustig', 'zacks', 'borella',
    'carretti', 'cornoldi', 'passolunghi', 'swanson',
    'mikels', 'rudner', 'masson', 'menon', 'atkinson', 'braver',
    'monsell', 'navon', 'posner', 'treisman', 'lavie', 'desimone',
    'barch', 'botvinick', 'cohen', 'miller', 'todd', 'marois',

    # Common affiliation/institutional words
    'department', 'university', 'college', 'institute', 'school', 'center',
    'laboratory', 'centre', 'faculty', 'division', 'hospital', 'clinic',
    'science', 'sciences', 'medicine', 'psychology', 'neuroscience', 'biology'
}


def preprocess_text(text):
    """
    Text preprocessing for topic modeling:
    - Rejoin hyphenated line breaks from PDF extraction
    - Remove special characters, numbers, URLs, emails
    - Fix common OCR errors and PDF artifacts
    - Convert to lowercase and filter short words
    - Lemmatize words and remove stopwords
    """
    # Rejoin hyphenated line breaks (e.g., "signi-\nficant" -> "significant")
    # This is critical for PDF-extracted text where words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()

    if len(text) < 50:
        return ""

    tokens = text.split()
    try:
        english_stopwords = set(stopwords.words("english"))
        english_stopwords.update(ENHANCED_STOPWORDS)
    except Exception:
        english_stopwords = set(ENHANCED_STOPWORDS)

    english_stopwords.update({
        'was', 'has', 'had', 'were', 'been', 'did', 'does', 'got',
        'set', 'get', 'put', 'let', 'say', 'see', 'saw', 'run', 'ran',
        'non', 'per', 'via', 'yet', 'far', 'may', 'also', 'well',
        'www', 'http', 'https', 'org', 'com', 'edu', 'pdf', 'pmc',
        'fig', 'tab', 'vol', 'ref', 'ect', 'etc',
    })

    # Stem all stopwords too, so stemmed tokens get caught
    if stemmer:
        english_stopwords = {stemmer.stem(w) for w in english_stopwords} | english_stopwords

    # Common word fragments from broken PDF hyphenation that slip through
    _FRAGMENTS = {
        'signi', 'cant', 'cantly', 'cation', 'ficant', 'tion', 'tial',
        'ment', 'ect', 'ous', 'ity', 'ive', 'ful', 'ness', 'ing',
        'ble', 'ally', 'ical', 'ated', 'ting', 'tions', 'ments',
        'sion', 'ence', 'ance', 'able', 'ible',
        'wa', 'ha', 'tho', 'thu', 'whi', 'ther',  # lemmatizer artifacts
        'com', 'atr', 'lar', 'ter', 'pre', 'pro', 'dis', 'sub',  # common fragments
        'speci', 'neurosci', 'lett', 'dif',  # truncated words from PDFs
        'tnum', 'cits', 'erences', 'gural', 'ciency', 'schizophr',  # PDF artifacts
        'psychol', 'cogn', 'percept', 'behav', 'psychiatr', 'pmid',  # journal abbrevs
        'buff', 'camo', 'numer', 'cognit', 'cereb', 'proce',  # fragments
    }

    # Known 3-letter scientific acronyms worth keeping
    _VALID_SHORT = {
        'eeg', 'erp', 'meg', 'roi', 'fmr', 'pet', 'mri', 'dti', 'tms',
        'pfc', 'acc', 'ica', 'svm', 'bci', 'emg', 'lfp', 'snr', 'bdd',
        'vep', 'eog', 'ecg', 'roc', 'auc', 'map', 'fly', 'eye',
        'arm', 'leg', 'ear', 'jaw', 'lip', 'rib', 'hip', 'gut',  # body parts
    }

    filtered_tokens = []
    for word in tokens:
        if not word.isalpha():
            continue

        # For 3-char words: only keep known scientific acronyms
        if len(word) <= 3:
            if word in _VALID_SHORT:
                filtered_tokens.append(word)
            continue

        # Skip likely word fragments (short tokens that look like suffixes)
        if word in _FRAGMENTS:
            continue

        # Snowball stemmer handles both inflectional AND derivational morphology:
        # "emotional"/"emotion" → "emot", "behavioral"/"behavior" → "behavior"
        # "neuronal"/"neuron" → "neuron", "oscillatory"/"oscillation" → "oscil"
        if stemmer:
            stem = stemmer.stem(word)
        else:
            stem = word

        if stem not in english_stopwords and len(stem) >= 3:
            filtered_tokens.append(stem)

    return " ".join(filtered_tokens)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with encryption handling."""
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
                text += preprocess_text(page_text)
    except Exception as e:
        logger.error("Error processing %s: %s", pdf_path, str(e))
        return ""
    return text


def find_cutoff_position(text):
    """Find the position of the references/bibliography section."""
    pattern = r"\b(references|bibliography|works cited|literature cited|" \
              r"acknowledgments?|notes|endnotes|sources|cited works|" \
              r"bibliographie|referencias|literatur)\b"
    lower_text = text.lower()
    matches = [match.start() for match in re.finditer(pattern, lower_text)]

    if not matches:
        return len(text)

    cutoff_threshold = int(len(text) * 0.4)
    valid_matches = [pos for pos in matches if pos >= cutoff_threshold]

    if valid_matches:
        return min(valid_matches)

    if matches:
        last_match = max(matches)
        if last_match >= int(len(text) * 0.3):
            return last_match

    return len(text)


def clean_pdf_text(text):
    """Remove citations and bibliography from PDF text."""
    # Remove in-text citations
    text = re.sub(r'\([A-Z][a-zA-Z]+(?:\s+(?:et al\.|and|&)\s+[A-Z][a-zA-Z]+)*,?\s*\d{4}[a-z]?\)', ' ', text)
    text = re.sub(r'\[[0-9,\-\s]+\]', ' ', text)
    text = re.sub(r'\b[A-Z][a-zA-Z]+\s+(?:et al\.\s*)?\(\d{4}\)', ' ', text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = text.lower()

    cutoff_pos = find_cutoff_position(text)

    if cutoff_pos < len(text) * 0.9:
        return text[:cutoff_pos]
    else:
        return text


def prepare_gensim_corpus(doc_term_matrix, vectorizer):
    """Convert document-term matrix to Gensim corpus format."""
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
    return filename.replace(".pdf", ""), None, filename.replace(".pdf", "")


def get_all_stopwords(additional_stopwords=None):
    """Build the complete stopwords list from NLTK + enhanced + user-provided."""
    try:
        default_stopwords = stopwords.words("english")
    except Exception:
        default_stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at",
                             "to", "for", "of", "with", "by", "is", "are", "was", "were"]

    additional = additional_stopwords or []
    return list(set(default_stopwords + additional) | ENHANCED_STOPWORDS)

# PDF Topic Explorer

Topic modeling (LDA / HDP) for academic research papers. Upload a ZIP of PDFs, get topics, visualizations, and cross-corpus comparison.

---

## Lab Computer Setup (Windows, From Scratch)

### Step 1: Install Git
1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Use all default settings, click Next through everything
4. When done, open **Git Bash** (search for it in Start menu - use this for all commands below)

### Step 2: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12
3. **IMPORTANT: Check the box "Add Python to PATH"** before clicking Install
4. Verify in Git Bash: `python --version`

### Step 3: Install Node.js
1. Go to https://nodejs.org/
2. Download the **LTS** version
3. Run the installer with default settings
4. Verify in Git Bash: `node --version`

### Step 4: Clone and Set Up
Open **Git Bash** and run these one by one:

```bash
git clone https://github.com/mugdha2626/TopicModeling.git
cd TopicModeling/lda

# Install frontend
npm install

# Set up Python environment
python -m venv venv
source venv/Scripts/activate

# Install Python packages
pip install -r requirements.txt

# Download language data (one time only)
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

### Step 5: Run the App
```bash
npm start
```

Open `http://localhost:3000` in your browser.

> **If `npm start` doesn't work on Windows**, run frontend and backend in two separate Git Bash windows:
> - Window 1: `source venv/Scripts/activate && python src/app.py`
> - Window 2: `npm run start-frontend`

> **On macOS or Linux**, activate the virtual environment with `source venv/bin/activate` (instead of `source venv/Scripts/activate`) in the steps above. Everything else is the same.

---

## How to Use

1. Open `http://localhost:3000`
2. Upload a **ZIP file** containing PDF research papers
3. Pick **LDA** (you choose number of topics) or **HDP** (auto-discovers topics)
4. Click **Analyze** and wait for results
5. View topics, charts, and top papers per topic

### PDF Naming Format
For best metadata extraction, name your PDFs like:
```
Author et al. - 2023 - Title of the Paper.pdf
```

### Comparing Two Corpora
1. Run analysis on corpus A, download the topic-word CSV from results
2. Run analysis on corpus B, download the topic-word CSV from results
3. Go to the **Compare** tab, upload both CSVs
4. See OT distance, TVD heatmap, and best-match network

---

## Building a Corpus (`fetch_papers.py`)

Don't have PDFs yet? `fetch_papers.py` bulk-downloads full-text PDFs for a
PubMed search, so you can build your own corpus to feed the app.

It pulls **only** from two free, sanctioned open-access sources and rate-limits
itself to stay within each service's usage policy:

1. **PMC Open Access Subset** - NCBI's world-readable AWS bucket (`pmc-oa-opendata`), for papers with a PMCID
2. **Unpaywall** - for everything else that has a legally posted free copy (by DOI)

Anything neither source can get is written to `still_missing.csv` for your
library's interlibrary-loan service. Paywalled papers are never bypassed.

### Step 1: Export a CSV from PubMed
1. Search [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for your topic
2. **Save** → Selection: *All results* → Format: **CSV** → **Create file**
3. The CSV must contain `PMID`, `PMCID`, and `DOI` columns (PubMed's default export does)

### Step 2: Install dependencies
```bash
pip install requests pandas
```

### Step 3: Test with a few papers first
```bash
python fetch_papers.py --csv your-search.csv --email you@university.edu --limit 25
```
Use a **real email** - Unpaywall requires it and NCBI asks for it. A minute or
so later, check that PDFs appeared in the `pdfs/` folder.

### Step 4: Run the full fetch
```bash
python fetch_papers.py --csv your-search.csv --email you@university.edu --zip
```
This downloads everything available and, with `--zip`, bundles the PDFs into
`pdfs.zip` - ready to upload straight into the app.

**Options**
| Flag | What it does |
|------|--------------|
| `--csv` | PubMed CSV export (required) |
| `--email` | Your email, required by Unpaywall/NCBI (required) |
| `--limit N` | Stop after N new PDFs per stage - good for a quick test |
| `--stage {1,2}` | Run only PMC (1) or only Unpaywall (2) |
| `--zip` | Zip the PDFs when finished |
| `--out DIR` | Output folder (default `pdfs/`) |

**Resumable:** progress is saved in `pdfs/_state.json`. Press `Ctrl-C` and rerun
the same command any time to pick up where it left off - already-downloaded
papers are skipped.

> Not every paper is free to download - only a fraction of academic articles are
> open access - so expect to retrieve a subset of your search, not all of it.

---

## Project Structure
```
TopicModeling/
├── fetch_papers.py             # Bulk-download open-access PDFs from a PubMed CSV
└── lda/
    ├── src/
    │   ├── App.js              # React frontend
    │   ├── app.py              # Flask backend (LDA + HDP)
    │   ├── preprocessing.py    # PDF text extraction + cleaning
    │   ├── comparison_utils.py # TVD, Optimal Transport, stats
    │   └── test_models.py      # Validation tests
    ├── public/                 # Static assets
    ├── package.json            # Node dependencies + scripts
    └── requirements.txt        # Python dependencies
```

## BERTopic Experiments (neural embedding topic modeling)

Alongside LDA / HDP, `lda/test_bertopic.py` runs **BERTopic** (Grootendorst 2022) on the same
PDF corpora: sentence-transformer embeddings → UMAP → HDBSCAN → class-based TF-IDF. The web app is
unchanged; this is a script-level experiment. Full explanation, parameters and citations:
`lda/docs/BERTOPIC_STUDY_GUIDE.md`; results: `lda/docs/BERTOPIC_RESULTS.md`.

```bash
cd lda
pip install -r requirements.txt            # adds bertopic, sentence-transformers, umap-learn, hdbscan, torch

# 100 + 1000 papers, MiniLM + PubMedBERT encoders, natural topics + 5-topic reduced run
python test_bertopic.py
python test_bertopic.py --all              # + 10 000 papers (embedding cached after the first pass)
python test_bertopic.py --models minilm --corpus path/to/papers.zip --label "my corpus"

# LDA vs HDP vs BERTopic on one table (also writes each model's topic-word CSVs)
python test_models_all.py --lda-k 5

# TVD / optimal-transport comparisons: BERTopic across corpus sizes, and vs LDA / HDP
python compare_bertopic.py --all

# "How many papers are enough?": agreement of the nested 100 ⊂ 1k ⊂ 10k runs
# (topic-embedding recovery, TVD coverage, ARI/NMI on shared papers) and a
# saturation curve over nested random subsets of the 10k corpus
python saturation_bertopic.py --given
python saturation_bertopic.py --curve --sizes 100,250,500,1000,2000,4000
```

First run downloads two free Hugging Face models (~90 MB `all-MiniLM-L6-v2`, ~440 MB
`NeuML/pubmedbert-base-embeddings`) into `~/.cache/huggingface`. On Apple Silicon the encoders run
on the GPU via PyTorch `mps`; elsewhere they fall back to CPU. Raw texts, chunks and embeddings are
cached under `lda/cache/<corpus>/` so re-runs and parameter changes are fast. Outputs land in
`lda/results/bertopic/<corpus>_<model>/{natural,k5}/` (`topics.md`, `metrics.json`, CSVs in the
app's Compare-tab format, PNG charts, Plotly HTML). Add `--min-cluster-size N` to override the
topic-granularity heuristic.

## Troubleshooting

**Port 5001 in use:** In Git Bash: `netstat -ano | findstr :5001` then `taskkill /PID <the_pid> /F`

**Python module not found:** Make sure venv is activated: `source venv/Scripts/activate`

**NLTK data missing:** `python -c "import nltk; nltk.download('all')"`

**Node issues:** `rm -rf node_modules && npm install`

## System Requirements
- RAM: 8GB minimum, 16GB recommended for large corpora
- Storage: 2GB free space

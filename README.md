# PDF Topic Explorer

Topic modeling (LDA / HDP) for academic research papers. Upload a ZIP of PDFs, get topics, visualizations, and cross-corpus comparison.

---

## Lab Computer Setup (Windows, From Scratch)

### Step 1: Install Git
1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Use all default settings, click Next through everything
4. When done, open **Git Bash** (search for it in Start menu — use this for all commands below)

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

## Project Structure
```
TopicModeling/
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

## Troubleshooting

**Port 5001 in use:** In Git Bash: `netstat -ano | findstr :5001` then `taskkill /PID <the_pid> /F`

**Python module not found:** Make sure venv is activated: `source venv/Scripts/activate`

**NLTK data missing:** `python -c "import nltk; nltk.download('all')"`

**Node issues:** `rm -rf node_modules && npm install`

## System Requirements
- RAM: 8GB minimum, 16GB recommended for large corpora
- Storage: 2GB free space

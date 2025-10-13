# Local Setup Instructions

## Prerequisites

Before you begin, make sure you have the following installed on your system:

### Required Software:
1. **Node.js** (version 14 or higher)
   - Download from [nodejs.org](https://nodejs.org/)
   - Verify installation: `node --version` and `npm --version`

2. **Python** (version 3.8 or higher)
   - Download from [python.org](https://python.org/)
   - Verify installation: `python3 --version` or `python --version`

3. **Git**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation: `git --version`

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/mugdha2626/TopicModeling.git
cd TopicModeling/lda
```

### Step 2: Install Frontend Dependencies
```bash
npm install
```

### Step 3: Install Python Dependencies
```bash
# Create a virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
The application requires NLTK data. Run this Python script once:
```bash
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('NLTK data downloaded successfully!')
"
```

## Running the Application

### Option 1: Run Both Frontend and Backend Together (Recommended)
```bash
npm start
```
This command will:
- Start the Python Flask backend on `http://localhost:5001`
- Start the React frontend on `http://localhost:3000`
- Automatically open your browser to the application

### Option 2: Run Frontend and Backend Separately

#### Terminal 1 - Start Backend:
```bash
npm run start-backend
# OR manually:
# python3 src/app.py
```

#### Terminal 2 - Start Frontend:
```bash
npm run start-frontend
# OR manually:
# npm start
```

## Using the Application

1. **Open your browser** to `http://localhost:3000`
2. **Upload a ZIP file** containing PDF research papers
3. **Configure analysis settings:**
   - Number of topics (for LDA)
   - Number of words per topic
   - Additional stopwords (optional)
4. **Choose model type:** LDA or HDP
5. **Click "Analyze"** and wait for results
6. **View results:** Topic charts, word distributions, and top papers

## File Format Requirements

### PDF Files in ZIP:
Your ZIP file should contain PDF files named in this format:
```
Author et al. - Year - Title.pdf
```
Example:
```
Smith et al. - 2023 - Machine Learning in Healthcare.pdf
Johnson and Brown - 2022 - Climate Change Analysis.pdf
```

### Comparison Feature:
If using the topic comparison feature, you'll need CSV files with this structure:
```csv
Topic_ID,Topic_Name,Word,Probability
0,Topic 1,word1,0.1234
0,Topic 1,word2,0.0987
1,Topic 2,word3,0.1456
```

## Troubleshooting

### Common Issues:

#### 1. Port Already in Use
If you see "Port 5001 already in use":
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9
# OR change port in src/app.py
```

#### 2. Python Module Not Found
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 3. NLTK Data Missing
```bash
python3 -c "
import nltk
nltk.download('all')
"
```

#### 4. Node/NPM Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 5. CORS Issues
If you see CORS errors in the browser console:
- Make sure both frontend (port 3000) and backend (port 5001) are running
- Check that the API URLs in the frontend are pointing to `localhost:5001`

## System Requirements

- **RAM**: Minimum 8GB (16GB recommended for large document sets)
- **Storage**: At least 2GB free space
- **CPU**: Multi-core processor recommended for faster analysis

## Development Notes

### Project Structure:
```
TopicModeling/
├── lda/
│   ├── src/
│   │   ├── App.js          # React frontend
│   │   ├── app.py          # Flask backend
│   │   └── comparison_utils.py
│   ├── public/
│   ├── build/              # Production build
│   ├── package.json        # Node dependencies
│   ├── requirements.txt    # Python dependencies
│   └── vercel.json         # Deployment config
└── README.md
```

### Environment Variables:
- `REACT_APP_API_URL`: Backend URL (defaults to `http://localhost:5001`)

### Available Scripts:
- `npm start`: Run both frontend and backend
- `npm run start-frontend`: Run only React app
- `npm run start-backend`: Run only Python API
- `npm run build`: Build production frontend
- `npm test`: Run tests

## Getting Help

If you encounter issues:
1. Check the terminal output for error messages
2. Ensure all prerequisites are installed
3. Verify that both ports 3000 and 5001 are available
4. Check the browser console for frontend errors
5. Review the Python console for backend errors

## Features

- **LDA Topic Modeling**: Traditional Latent Dirichlet Allocation
- **HDP Topic Modeling**: Hierarchical Dirichlet Process (automatic topic discovery)
- **Interactive Visualizations**: Bar charts, word clouds, topic distributions
- **Topic Comparison**: Compare topic models between different document sets
- **Paper Analysis**: Identify top papers for each topic
- **Decade Analysis**: Track topic evolution over time periods
- **Export Functionality**: Download results and visualizations

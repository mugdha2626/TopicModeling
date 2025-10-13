# Deployment Guide for Topic Modeling App

## Overview
This application consists of:
- **Frontend**: React app (deployed to Vercel)
- **Backend**: Python Flask API (needs separate hosting)

## Frontend Deployment to Vercel

### Prerequisites
1. Install Vercel CLI: `npm i -g vercel`
2. Create a Vercel account at [vercel.com](https://vercel.com)

### Step 1: Deploy to Vercel
```bash
cd lda
vercel
```

Follow the prompts:
- Set up and deploy? **Y**
- Which scope? Choose your account
- Link to existing project? **N** (for first deployment)
- Project name: `topic-modeling-app` (or your preferred name)
- Directory: `./` (current directory)
- Override settings? **N**

### Step 2: Set Environment Variables
After deployment, set the production API URL:
```bash
vercel env add REACT_APP_API_URL
```
Enter your backend URL when prompted (e.g., `https://your-api.herokuapp.com`)

### Step 3: Redeploy
```bash
vercel --prod
```

## Backend Deployment Options

Since Vercel doesn't support Python Flask apps directly, deploy your backend separately:

### Option 1: Heroku (Recommended)
1. Create `requirements.txt` in your root directory (already exists)
2. Create `Procfile`:
   ```
   web: gunicorn --chdir src app:app
   ```
3. Install gunicorn: `pip install gunicorn`
4. Deploy to Heroku:
   ```bash
   heroku create your-topic-api
   git push heroku main
   ```

### Option 2: Railway
1. Connect your GitHub repo to Railway
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python src/app.py`

### Option 3: Render
1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python src/app.py`

## Environment Variables Setup

### Frontend (.env.production)
```
REACT_APP_API_URL=https://your-backend-url.com
```

### Backend
Set these on your hosting platform:
```
FLASK_ENV=production
PORT=5001
```

## Local Development
```bash
# Install dependencies
npm install

# Start development (both frontend and backend)
npm start

# Or start separately
npm run start-frontend  # React on :3000
npm run start-backend   # Flask on :5001
```

## Troubleshooting

### CORS Issues
If you get CORS errors, ensure your backend includes:
```python
from flask_cors import CORS
CORS(app)
```

### Build Issues
- Ensure all dependencies are in package.json
- Check that REACT_APP_API_URL is set correctly
- Verify the build output in the `build/` directory

### API Connection Issues
- Check that backend is deployed and accessible
- Verify REACT_APP_API_URL matches your backend URL
- Check browser network tab for failed requests

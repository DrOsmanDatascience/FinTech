# 🚀 Deployment Guide

This guide covers deploying the Stock PCA Cluster Analysis app via GitHub and Streamlit Community Cloud.

## Prerequisites

- GitHub account
- (Optional) OpenAI API key for the chatbot feature
- (Optional) Streamlit Community Cloud account (free)

---

## Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to [github.com/new](https://github.com/new)
2. Enter repository name: `stock-pca-analysis` (or your preferred name)
3. Set to **Public** (required for free Streamlit Cloud hosting)
4. Click **Create repository**

### Option B: Using Git Command Line

```bash
# Navigate to your project folder
cd stock_pca_app

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Stock PCA Cluster Analysis app"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/stock-pca-analysis.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 2: Project Structure for GitHub

Ensure your repository has this structure:

```
stock-pca-analysis/
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualizations.py
│   └── chatbot.py
├── data/
│   └── README.md            # Data format documentation
├── .gitignore
├── app.py                   # Main application (entry point)
├── config.py
├── requirements.txt
├── README.md
└── DEPLOYMENT.md            # This file
```

**Important:** Make sure `app.py` is in the root directory!

---

## Step 3: Deploy to Streamlit Community Cloud

### 3.1 Sign Up / Log In

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **Sign in with GitHub**
3. Authorize Streamlit to access your repositories

### 3.2 Deploy Your App

1. Click **New app** button
2. Select your repository: `YOUR_USERNAME/stock-pca-analysis`
3. Select branch: `main`
4. Set Main file path: `app.py`
5. Click **Deploy!**

### 3.3 Configure Secrets (for OpenAI Chatbot)

After deployment, add your API keys:

1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
2. Click the **⋮** menu → **Settings**
3. Click **Secrets** in the left sidebar
4. Add your secrets in TOML format:

```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```

5. Click **Save**
6. Reboot your app for changes to take effect

---

## Step 4: Using Your Own Data

### Option A: Include Data in Repository

1. Add your CSV files to the `data/` folder
2. Update `.gitignore` to NOT ignore your data files:

```gitignore
# Comment out or remove this line:
# data/*.csv
```

3. Update `utils/data_loader.py`:

```python
def load_stock_data(use_sample=False):  # Change to False
    file_path = os.path.join(DATA_DIR, 'stock_data.csv')
    return pd.read_csv(file_path)
```

### Option B: Load Data from External URL

1. Host your data files elsewhere (e.g., another GitHub repo, S3, etc.)
2. Update the GitHub URL in `utils/data_loader.py`:

```python
GITHUB_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_DATA_REPO/main/data"
```

---

## Step 5: Custom Domain (Optional)

Streamlit Cloud provides a URL like:
`https://your-app-name.streamlit.app`

For a custom domain:
1. Upgrade to Streamlit Teams (paid)
2. Or use a URL redirect service

---

## Updating Your Deployed App

Any push to your GitHub repository will automatically trigger a redeployment:

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main
```

The app will automatically redeploy within a few minutes.

---

## Troubleshooting

### "ModuleNotFoundError"
- Ensure all dependencies are in `requirements.txt`
- Check that module names are spelled correctly

### "App not loading"
- Check the Streamlit Cloud logs (click **Manage app** → **Logs**)
- Verify `app.py` is in the repository root

### "Secrets not working"
- Ensure TOML syntax is correct (strings need quotes)
- Reboot the app after adding secrets

### "Data not loading"
- Check file paths are relative to the repository root
- Verify CSV files are committed to the repository

---

## Resource Limits (Free Tier)

Streamlit Community Cloud free tier includes:
- 1 GB memory
- Public apps only
- Apps sleep after inactivity (wake on visit)

For production use with higher limits, consider:
- Streamlit Teams (paid)
- Self-hosting on AWS/GCP/Azure
- Deploying on Heroku, Railway, or Render

---

## Quick Reference Commands

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/stock-pca-analysis.git

# Create branch for new features
git checkout -b feature/new-visualization

# Push changes
git add .
git commit -m "Add new feature"
git push origin feature/new-visualization

# Merge to main (triggers deployment)
git checkout main
git merge feature/new-visualization
git push origin main
```

---

## Support

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/YOUR_USERNAME/stock-pca-analysis/issues)

# 🚀 Setup Instructions for DrOsmanDatascience/FinTech

Follow these steps to add the Stock PCA Analysis app to your existing FinTech repository.

---

## Option A: Add as a Subfolder (Recommended)

### Step 1: Clone Your Repository
```bash
git clone https://github.com/DrOsmanDatascience/FinTech.git
cd FinTech
```

### Step 2: Create the App Folder
```bash
mkdir -p stock_pca_app
```

### Step 3: Copy App Files
Extract the downloaded ZIP and copy all files into the `stock_pca_app` folder:

```
FinTech/
├── (your existing files)
├── stock_pca_app/           # NEW - Add this folder
│   ├── .streamlit/
│   │   └── config.toml
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── visualizations.py
│   │   └── chatbot.py
│   ├── data/
│   │   └── README.md
│   ├── app.py
│   ├── config.py
│   └── requirements.txt
└── README.md
```

### Step 4: Push to GitHub
```bash
git add .
git commit -m "Add Stock PCA Cluster Analysis app"
git push origin main
```

### Step 5: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select:
   - Repository: `DrOsmanDatascience/FinTech`
   - Branch: `main`
   - Main file path: `stock_pca_app/app.py`
4. Click **Deploy!**

---

## Option B: Deploy from Root (Alternative)

If you want the app at the root level of your repo:

### Step 1: Copy Files to Root
Copy all app files directly to your FinTech repository root:

```
FinTech/
├── .streamlit/
│   └── config.toml
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualizations.py
│   └── chatbot.py
├── data/
│   └── (your PCA data files)
├── app.py                    # Main entry point
├── config.py
├── requirements.txt
└── (your existing files)
```

### Step 2: Deploy
On Streamlit Cloud, set Main file path to: `app.py`

---

## Adding Your Real Data

### Required CSV Files

Place these in the `data/` folder (or configure the path):

**1. stock_data.csv**
```csv
ticker,permno,company_name,sector,market_weight,value_score,quality_score,fin_strength_score,momentum_score,volatility_score,liquidity_score
AAPL,14593,Apple Inc,Technology,7.1,0.65,0.85,0.78,0.72,0.35,0.95
CL,19174,Colgate-Palmolive,Consumer Staples,0.45,0.55,0.82,0.75,0.58,0.28,0.72
...
```

**2. pca_results.csv**
```csv
ticker,permno,company_name,pc1,pc2,cluster
AAPL,14593,Apple Inc,2.5,0.8,0
CL,19174,Colgate-Palmolive,2.47,-0.56,2
...
```

**3. time_series_pca.csv** (optional, for animations)
```csv
date,ticker,pc1,pc2
2024-01-31,AAPL,2.3,0.7
2024-02-29,AAPL,2.4,0.75
...
```

### Enable Real Data Loading

Edit `utils/data_loader.py` and change:
```python
def load_stock_data(use_sample=False):  # Change True to False
```

---

## Configure OpenAI Chatbot (Optional)

### On Streamlit Cloud:
1. Go to your deployed app
2. Click **Settings** → **Secrets**
3. Add:
```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

### For Local Development:
Create a `.env` file (don't commit this!):
```
OPENAI_API_KEY=sk-your-api-key-here
```

---

## Quick Reference

| Action | Command/Location |
|--------|------------------|
| Your repo | `https://github.com/DrOsmanDatascience/FinTech` |
| Deploy URL | `https://share.streamlit.io` |
| Main file | `stock_pca_app/app.py` or `app.py` |
| Add secrets | App Settings → Secrets |

---

## Troubleshooting

**App not loading?**
- Check that `requirements.txt` is in the same directory as `app.py`
- Verify the Main file path in Streamlit Cloud settings

**Data not showing?**
- Ensure CSV files are committed to the repository
- Check file paths in `data_loader.py`

**Chatbot not working?**
- Add `OPENAI_API_KEY` in Streamlit Cloud Secrets
- The app works without it (uses fallback responses)

---

Need help? The app will work immediately with sample data - you can add real data later!

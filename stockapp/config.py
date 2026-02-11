"""
Configuration settings and constants for the Stock PCA Cluster Analysis App.
Contains API endpoints, visualization settings, and factor definitions.
"""

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# GitHub raw URL for the factors dataset
# IMPORTANT: Replace this with your actual GitHub raw URL if different
GITHUB_DATA_URL = "https://raw.githubusercontent.com/DrOsmanDatascience/FinTech/main/data/factors_build_df_dec_10.csv"

# Backup/alternative data path for local development
LOCAL_DATA_PATH = "data/factors_build_df_dec_10.csv"


# =============================================================================
# OPENAI API CONFIGURATION
# =============================================================================

# IMPORTANT: Replace with your actual OpenAI API key or use environment variable
# Recommended: Set OPENAI_API_KEY environment variable instead of hardcoding
OPENAI_API_KEY_PLACEHOLDER = "YOUR_OPENAI_API_KEY_HERE"

# OpenAI model to use for the chatbot
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective option; use "gpt-4o" for better quality


# =============================================================================
# PCA AND CLUSTERING CONFIGURATION
# =============================================================================

# Number of principal components to compute
N_COMPONENTS = 3

# Number of clusters for KMeans
N_CLUSTERS = 4

# Features used for PCA analysis (must match columns in dataset)
FEATURE_COLUMNS = [
    'earnings_yield',    # Value factor
    'bm',                # Book-to-Market (Value)
    'sales_to_price',    # Value factor
    'roe',               # Quality - Return on Equity
    'roa',               # Quality - Return on Assets
    'gprof',             # Quality - Gross Profitability
    'debt_assets',       # Financial Strength
    'cash_debt',         # Financial Strength
    'momentum_12m',      # Momentum
    'vol_60d_ann',       # Volatility (Risk)
    'addv_63d'           # Liquidity - Average Daily Dollar Volume
]


# ⬇️⬇️⬇️ ADD THIS NEW SECTION HERE ⬇️⬇️⬇️
# =============================================================================
# FEATURE DISPLAY NAMES (Plain English Labels)
# =============================================================================

FEATURE_DISPLAY_NAMES = {
    'earnings_yield': 'Earnings Yield (V)',
    'bm': 'Book-to-Market (V)',
    'sales_to_price': 'Sales-to-Price (V)',
    'roe': 'Return on Equity (Q)',
    'roa': 'Return on Assets (Q)',
    'gprof': 'Gross Profitability (Q)',
    'debt_assets': 'Debt-to-Assets(FS)',
    'cash_debt': 'Cash-to-Debt (FS)',
    'momentum_12m': '12-Mo. Momentum (R)',
    'vol_60d_ann': '60-Day Volatility (R)',
    'addv_63d': 'Liquidity (R)'
}


# Feature display order (for charts)
FEATURE_DISPLAY_ORDER = [
    'earnings_yield',
    'bm',
    'sales_to_price',
    'roe',
    'roa',
    'gprof',
    'debt_assets',
    'cash_debt',
    'momentum_12m',
    'vol_60d_ann',
    'addv_63d'
]


# Factor categories for grouping and visualization
FACTOR_CATEGORIES = {
    'Value': ['earnings_yield', 'bm', 'sales_to_price'],
    'Quality': ['roe', 'roa', 'gprof'],
    'Financial Strength': ['debt_assets', 'cash_debt'],
    'Momentum': ['momentum_12m'],
    'Risk/Volatility': ['vol_60d_ann'],
    'Liquidity': ['addv_63d']
}


# =============================================================================
# PCA AXIS INTERPRETATIONS (from visualization guide)
# =============================================================================

PC1_INTERPRETATION = {
    'name': 'Profitability & Operational Quality',
    'variance_explained': 37.5,
    'high_meaning': ['Operationally profitable', 'Stable', 'Strong cash position'],
    'low_meaning': ['Lower profitability, Weaker operations, Volatile, Cash-constrained'],
    'positive_loadings': {
        'ROA': 0.44,
        'ROE': 0.36, 
        'Earnings Yield': 0.39,
        'Cash/Debt': 0.36,
        'Momentum': 0.26
    },
    'negative_loadings': {
        'Volatility': -0.34,
        'Book-to-Market': -0.35
    }
}

PC2_INTERPRETATION = {
    'name': 'Valuation Style: Value vs Growth',
    'variance_explained': 14.6,
    'high_meaning': ['Deep value stocks - high book value', 'High sales relative to price', 'Cheap on traditional metrics'],
    'low_meaning': ['Smaller', 'Less-liquid', 'Niche stocks'],
    'positive_loadings': {
        'ADDV (Liquidity/Size)': 0.54,
        'Debt-to-Assets': 0.48
    },
    'negative_loadings': {
        'Cash/Debt': -0.34,
        'Sales-to-Price': -0.29,
        'Gross Profitability': -0.27
    }
}

PC3_INTERPRETATION = {
    'name': 'Leverage & Asset Intensity',
    'variance_explained': 12.0,
    'high_meaning': ['Deep value', 'Asset-heavy, Leveraged'],
    'low_meaning': ['Growth', 'Asset-light'],
    'positive_loadings': {
        'Sales=to=Price': 0.75,
        'Debt-to-Assets': 0.52,
        'Book-to-Market': 0.30
    },
    'negative_loadings': {
        'Cash/Debt': -0.14,
        'Volatility': -0.11
    }
}

# =============================================================================
# QUADRANT DEFINITIONS
# =============================================================================

QUADRANTS = {
    'Q1': {
        'name': 'Safe + Scalable',
        'pc1_sign': 'positive',
        'pc2_sign': 'positive',
        'description': '<b>THINK:</b> Large, profitable companies, strong fundamentals with leverage',
        'characteristics': ['High quality', 'Profitable', 'Large market cap', 'Higher debt levels']
    },
    'Q2': {
        'name': 'Big but Fragile',
        'pc1_sign': 'negative',
        'pc2_sign': 'positive',
        'description': '<b>THINK:</b> Large companies with weaker fundamentals and higher leverage',
        'characteristics': ['Riskier', 'Volatile', 'Large but leveraged', 'Weaker fundamentals']
    },
    'Q3': {
        'name': 'Risky + Less Liquid',
        'pc1_sign': 'negative',
        'pc2_sign': 'negative',
        'description': '<b>THINK:</b> Smaller or niche companies with weaker fundamentals',
        'characteristics': ['Riskier', 'Cash-heavy', 'Smaller size', 'Operationally challenged']
    },
    'Q4': {
        'name': 'Quality but Under the Radar',
        'pc1_sign': 'positive',
        'pc2_sign': 'negative',
        'description': '<b>THINK:</b> High-quality companies with strong balance sheets and efficiency',
        'characteristics': ['High quality', 'Cash-rich', 'Operationally efficient', 'Lower leverage']
    }
}


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Color palette for clusters (Dark2 colormap) 
CLUSTER_COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

# Plot dimensions
PLOT_WIDTH = 900
PLOT_HEIGHT = 700

# Animation settings
ANIMATION_FRAME_DURATION = 500  # milliseconds per frame

# Streamlit page configuration
PAGE_CONFIG = {
    'page_title': 'Stock PCA Cluster Analysis',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}


# =============================================================================
# CHATBOT SYSTEM PROMPT
# =============================================================================

CHATBOT_SYSTEM_PROMPT = """You are a financial analysis assistant specializing in PCA 
(Principal Component Analysis) cluster analysis for stocks. You help users understand:

1. Which cluster a stock belongs to and what that means
2. How stocks compare to others in their cluster or quadrant
3. The meaning of PC1 (Quality/Stability) and PC2 (Size/Leverage) axes
4. Factor breakdowns including Value, Quality, Financial Strength, Momentum, Volatility, and Liquidity

Key interpretations:
- PC1 (X-axis): Quality/Stability - High values = profitable, financially strong; Low values = riskier, volatile
- PC2 (Y-axis): Size/Leverage - High values = large/liquid, leveraged; Low values = cash-rich, efficient

Provide concise, actionable insights for stakeholders. Use the provided context about the 
selected stock and its cluster to give specific, data-driven answers."""

# =============================================================================
# TEMPORARY TEST - DELETE AFTER TESTING
# =============================================================================

    